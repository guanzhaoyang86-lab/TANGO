from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from ts_grounder.event_metrics import (
        DEFAULT_BOUNDARY_AWARE_REWARD_WEIGHTS,
        DEFAULT_MAX_INDEX,
        DEFAULT_PREDICTION_SCHEMA,
        DEFAULT_RESIDUAL_SAMPLING_RATIOS,
        DEFAULT_SERIES_LENGTH,
        DEFAULT_TAU_GOOD,
        DEFAULT_TAU_MATCH,
        build_balanced_residual_epoch,
        parse_sampling_ratios,
    )
    from ts_grounder.rl_reward import (
        DEFAULT_REWARD_WEIGHTS,
        compute_boundary_aware_grounder_reward_details,
        compute_grounder_reward_details,
        resolve_boundary_aware_reward_weights,
        resolve_reward_weights,
    )
    from ts_grounder.vlm_eval import load_jsonl
    from ts_grounder.vlm_prompting import build_chat_messages
    from ts_grounder.vlm_training import _clean_generated_text, _generate_predictions_for_records
    from ts_grounder.utils import load_yaml
    from ts_grounder.schema import GROUNDING_CONTRACT_VERSION
    from ts_grounder.supervision import annotation_scope, validate_generation_budget, validate_prepared_record
else:
    from .event_metrics import (
        DEFAULT_BOUNDARY_AWARE_REWARD_WEIGHTS,
        DEFAULT_MAX_INDEX,
        DEFAULT_PREDICTION_SCHEMA,
        DEFAULT_RESIDUAL_SAMPLING_RATIOS,
        DEFAULT_SERIES_LENGTH,
        DEFAULT_TAU_GOOD,
        DEFAULT_TAU_MATCH,
        build_balanced_residual_epoch,
        parse_sampling_ratios,
    )
    from .rl_reward import (
        DEFAULT_REWARD_WEIGHTS,
        compute_boundary_aware_grounder_reward_details,
        compute_grounder_reward_details,
        resolve_boundary_aware_reward_weights,
        resolve_reward_weights,
    )
    from .vlm_eval import load_jsonl
    from .vlm_prompting import build_chat_messages
    from .vlm_training import _clean_generated_text, _generate_predictions_for_records
    from .utils import load_yaml
    from .schema import GROUNDING_CONTRACT_VERSION
    from .supervision import annotation_scope, validate_generation_budget, validate_prepared_record


def _resolve_dtype(value: str | None):
    if value in (None, "", "none"):
        return None
    import torch

    normalized = value.strip().lower()
    dtype_map = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in dtype_map:
        raise ValueError(f"Unsupported torch dtype: {value}")
    return dtype_map[normalized]


def _model_device(model):
    import torch

    return next(
        (param.device for param in model.parameters() if param.device.type != "meta"),
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )


def _move_to_device(batch: dict[str, Any], device) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def _decode_new_tokens(processor, model_inputs: dict[str, Any], generated_ids) -> list[str]:
    input_ids = model_inputs.get("input_ids")
    if input_ids is None:
        return [
            _clean_generated_text(text)
            for text in processor.batch_decode(generated_ids, skip_special_tokens=True)
        ]
    prompt_length = int(input_ids.shape[1])
    new_token_ids = generated_ids[:, prompt_length:]
    if new_token_ids.shape[1] == 0:
        return ["" for _ in range(int(generated_ids.shape[0]))]
    return [
        _clean_generated_text(text)
        for text in processor.batch_decode(new_token_ids, skip_special_tokens=True)
    ]


def _resolve_image_path(record: dict[str, Any], image_root: str | None) -> Path:
    raw_path = record.get("image_path") or record.get("image_relpath")
    if raw_path is None:
        raise KeyError(f"record {record.get('id', '<unknown>')} is missing image_path")
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path
    if image_root:
        return Path(image_root).expanduser() / path
    return path


def _seq_len_for_record(record: dict[str, Any], ground_truth: Any) -> int:
    for key in ("seq_len", "series_length"):
        if key in record:
            return int(record[key])
    metadata = record.get("metadata", {})
    if isinstance(metadata, dict) and metadata.get("series_length") is not None:
        return int(metadata["series_length"])
    if isinstance(ground_truth, dict):
        for key in ("point_labels", "labels"):
            labels = ground_truth.get(key)
            if isinstance(labels, list):
                return len(labels)
    if isinstance(ground_truth, list):
        return len(ground_truth)
    raise KeyError(f"record {record.get('id', '<unknown>')} is missing seq_len")


def _ground_truth_for_record(record: dict[str, Any]) -> Any:
    if "target" in record and "ground_truth" in record and record["target"] != record["ground_truth"]:
        raise ValueError("Prepared target and ground_truth disagree; refusing alternate supervision")
    for key in ("ground_truth", "target"):
        if key in record:
            return record[key]
    raw_sample = record.get("raw_sample")
    if isinstance(raw_sample, dict):
        if raw_sample.get("events") is not None:
            return {"evidence": raw_sample.get("events", [])}
        if raw_sample.get("point_labels") is not None:
            return raw_sample["point_labels"]
    for key in ("point_labels", "labels"):
        if key in record:
            return record[key]
    raise KeyError(f"record {record.get('id', '<unknown>')} is missing ground truth")


def _forbidden_action_token_ids(model) -> list[int]:
    """Vision structure is valid in the prompt, never in a textual JSON action.

    A newly sampled image token has no corresponding image tensor. Allowing it
    can crash both incremental generation and full-sequence scoring. The same
    restriction is applied and normalized in both paths.
    """
    config = getattr(model, "config", None)
    forbidden = set()
    for container in (config, getattr(config, "text_config", None)):
        for name in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"):
            value = getattr(container, name, None)
            if value is not None:
                values = value if isinstance(value, (list, tuple)) else [value]
                forbidden.update(int(item) for item in values)
    return sorted(forbidden)


def _generate_candidates(
    *,
    processor,
    model,
    image,
    system_prompt: str,
    user_prompt: str,
    device,
    num_generations: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    import torch

    messages = build_chat_messages(system_prompt=system_prompt, user_prompt=user_prompt)
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = processor(text=[prompt_text], images=[image], return_tensors="pt")
    model_inputs = _move_to_device(model_inputs, device)

    tokenizer = getattr(processor, "tokenizer", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": True,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": 0,
        "typical_p": 1.0,
        "min_p": None,
        "epsilon_cutoff": 0.0,
        "eta_cutoff": 0.0,
        "num_beams": 1,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "min_length": 0,
        "min_new_tokens": 0,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "suppress_tokens": _forbidden_action_token_ids(model) or None,
        "begin_suppress_tokens": None,
        "bad_words_ids": None,
        "sequence_bias": None,
        "stop_strings": None,
        "watermarking_config": None,
        "guidance_scale": None,
        "token_healing": False,
        "exponential_decay_length_penalty": None,
        "renormalize_logits": False,
        "return_dict_in_generate": False,
        "num_return_sequences": int(num_generations),
    }
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = pad_token_id
    eos_token_id = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
    eos_ids = eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id]
    if eos_token_id is None or not eos_ids:
        raise ValueError("A known EOS token is required for correct response masking")
    if set(eos_ids) & set(generation_kwargs["suppress_tokens"] or []):
        raise ValueError("EOS cannot also be a forbidden vision-structure action token")

    model.eval()
    # no_grad (rather than inference_mode) keeps sampled IDs safe to use in a
    # subsequent differentiable gather without inference-tensor restrictions.
    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, **generation_kwargs)
    prompt_length = int(model_inputs["input_ids"].shape[1])
    if generated_ids.ndim != 2 or generated_ids.shape[0] != num_generations:
        raise RuntimeError("Generation returned an unexpected candidate batch")
    if not torch.equal(generated_ids[:, :prompt_length], model_inputs["input_ids"].expand(num_generations, -1)):
        raise RuntimeError("Only decoder-only VLM generation preserving the prompt prefix is supported")
    completion_mask = _completion_mask(generated_ids[:, prompt_length:], eos_token_id)
    # Reprocess the identical prompt for each candidate to expand packed visual
    # tensors correctly. Never decode/re-tokenize an action: that changes EOS,
    # whitespace and sometimes token boundaries, invalidating its likelihood.
    full_batch = processor(
        text=[prompt_text] * num_generations,
        images=[image] * num_generations,
        padding=True,
        return_tensors="pt",
    )
    full_batch = _move_to_device(full_batch, device)
    if not torch.equal(full_batch["input_ids"], generated_ids[:, :prompt_length]):
        raise RuntimeError("Batched prompt tokenization differs from generation tokenization")
    prompt_mask = full_batch.get("attention_mask", torch.ones_like(full_batch["input_ids"]))
    full_batch["input_ids"] = generated_ids
    full_batch["attention_mask"] = torch.cat((prompt_mask, completion_mask.to(prompt_mask.dtype)), dim=1)
    for key in ("token_type_ids", "mm_token_type_ids"):
        if key in full_batch:
            extension = torch.zeros_like(completion_mask, dtype=full_batch[key].dtype)
            full_batch[key] = torch.cat((full_batch[key], extension), dim=1)
    # VLMs calculate multimodal positions from the full sequence and grid data.
    full_batch.pop("position_ids", None)
    return {
        "texts": _decode_new_tokens(processor, model_inputs, generated_ids),
        "model_inputs": full_batch,
        "completion_mask": completion_mask,
        "prompt_length": prompt_length,
        "completion_ids": generated_ids[:, prompt_length:],
        "forbidden_action_token_ids": _forbidden_action_token_ids(model),
    }


def _completion_mask(completion_ids, eos_token_id):
    """Include the first EOS, exclude only positions after it (also PAD=EOS)."""
    import torch
    if completion_ids.ndim != 2 or completion_ids.shape[1] == 0:
        raise ValueError("At least one sampled response token is required")
    eos_ids = eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id]
    if not eos_ids:
        raise ValueError("At least one EOS token must be configured")
    is_eos = torch.zeros_like(completion_ids, dtype=torch.bool)
    for token_id in eos_ids:
        is_eos |= completion_ids == int(token_id)
    return (is_eos.long().cumsum(dim=1) - is_eos.long()) == 0


def _candidate_log_probs(*, model, rollout: dict[str, Any], temperature: float = 1.0):
    """Mean exact-action log likelihood under the temperature-scaled policy.

    This intentionally retains the paper implementation's sequence-mean
    objective; it is not PPO clipping or a conventional GRPO tokenwise KL.
    """
    import torch
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    full_batch = rollout["model_inputs"]
    prompt_length = int(rollout["prompt_length"])
    response_mask = rollout["completion_mask"]
    shift_labels = full_batch["input_ids"][:, prompt_length:]
    forbidden = rollout.get("forbidden_action_token_ids", _forbidden_action_token_ids(model))
    forbidden_tensor = None
    if forbidden:
        forbidden_tensor = torch.tensor(forbidden, device=shift_labels.device, dtype=torch.long)
        if bool((torch.isin(shift_labels, forbidden_tensor) & response_mask).any()):
            raise ValueError("A sampled JSON action contains a forbidden vision-structure token")
    outputs = model(**full_batch)
    # fp32 accumulation avoids bf16 overflow/cancellation for logsumexp and KL.
    shift_logits = outputs.logits[:, prompt_length - 1:-1, :].float() / temperature
    if shift_logits.shape[:2] != response_mask.shape:
        raise RuntimeError("Response logits and sampled token mask are misaligned")
    if forbidden:
        if min(forbidden) < 0 or max(forbidden) >= shift_logits.shape[-1]:
            raise ValueError("Vision token ID is outside the language vocabulary")
        forbidden_tensor = forbidden_tensor.to(shift_logits.device)
        shift_logits = shift_logits.index_fill(-1, forbidden_tensor, float("-inf"))
    target_logits = shift_logits.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    token_log_probs = target_logits - torch.logsumexp(shift_logits, dim=-1)
    lengths = response_mask.sum(dim=1)
    if not bool((lengths > 0).all()):
        raise RuntimeError("Empty response action mask")
    return token_log_probs.masked_fill(~response_mask, 0.0).sum(dim=1) / lengths


def _policy_objective(log_probs, advantages, ref_log_probs=None, *, reference_coef: float = 0.0):
    """The original sequence-mean policy loss plus squared-reference penalty."""
    import torch
    advantage_tensor = torch.as_tensor(advantages, device=log_probs.device, dtype=log_probs.dtype).detach()
    policy_loss = -(advantage_tensor * log_probs).mean()
    reference_penalty = torch.zeros((), device=log_probs.device, dtype=log_probs.dtype)
    sampled_log_ratio = torch.zeros_like(reference_penalty)
    if ref_log_probs is not None:
        reference = ref_log_probs.detach().to(device=log_probs.device, dtype=log_probs.dtype)
        log_ratio = log_probs - reference
        sampled_log_ratio = log_ratio.mean()
        reference_penalty = 0.5 * log_ratio.pow(2).mean()
    loss = policy_loss + float(reference_coef) * reference_penalty
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Non-finite RL objective; optimizer step aborted")
    return loss, policy_loss, reference_penalty, sampled_log_ratio


def _normalize_advantages(rewards: list[float]) -> list[float]:
    if not rewards:
        return []
    if not all(math.isfinite(float(reward)) for reward in rewards):
        raise ValueError("Rewards must be finite")
    mean_reward = sum(rewards) / len(rewards)
    variance = sum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards)
    std_reward = math.sqrt(variance)
    return [(reward - mean_reward) / (std_reward + 1e-6) for reward in rewards]


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _load_trainer_state(checkpoint_dir: str | Path | None) -> dict[str, Any]:
    if not checkpoint_dir:
        return {}
    state_path = Path(checkpoint_dir) / "trainer_state.json"
    if not state_path.exists():
        return {}
    return json.loads(state_path.read_text(encoding="utf-8"))


def _save_checkpoint(
    *,
    checkpoint_dir: Path,
    model,
    processor,
    optimizer,
    state: dict[str, Any],
    save_optimizer_state: bool,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(checkpoint_dir)
    (checkpoint_dir / "trainer_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if save_optimizer_state:
        import torch

        torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")


def _maybe_resume_optimizer(optimizer, checkpoint_dir: str | Path | None) -> bool:
    if not checkpoint_dir:
        return False
    optimizer_path = Path(checkpoint_dir) / "optimizer.pt"
    if not optimizer_path.exists():
        raise FileNotFoundError("Exact resume requires optimizer.pt; use a new run for weight-only warm starts")
    import torch

    optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu", weights_only=True))
    return True


def _build_optimizer(model, args):
    import torch
    from transformers import Adafactor

    optimizer_name = str(args.optimizer).lower()
    if optimizer_name == "adafactor":
        return Adafactor(
            model.parameters(),
            lr=float(args.learning_rate),
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
            weight_decay=float(args.weight_decay),
        )
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def _reward_weights_from_args(args) -> dict[str, float]:
    return resolve_reward_weights(
        {
            "event_f1": float(args.reward_event_f1_weight),
            "boundary_iou": float(args.reward_boundary_iou_weight),
            "hallucination_penalty": float(args.reward_hallucination_penalty_weight),
        }
    )


def _boundary_reward_weights_from_args(args) -> dict[str, float]:
    return resolve_boundary_aware_reward_weights(
        {
            "point": float(args.reward_point_weight),
            "event": float(args.reward_event_weight),
            "iou": float(args.reward_iou_weight),
            "boundary": float(args.reward_boundary_weight),
            "type": float(args.reward_type_weight),
        }
    )


def _next_position(epoch: int, record_index: int, num_records: int) -> tuple[int, int]:
    next_record_index = record_index + 1
    next_epoch = epoch
    if next_record_index >= num_records:
        next_epoch += 1
        next_record_index = 0
    return next_epoch, next_record_index


def _seed_step(seed: int, step: int) -> int:
    """Stateless per-step RNG makes checkpoint resume reproduce future rollouts."""
    import torch
    resolved = (int(seed) + 1000003 * int(step)) % (2**63 - 1)
    random.seed(resolved)
    torch.manual_seed(resolved)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved)
    return resolved


def _disable_dropout(model) -> int:
    import torch
    count = 0
    for module in model.modules():
        if isinstance(module, torch.nn.modules.dropout._DropoutNd) and module.p:
            module.p = 0.0
            count += 1
        # Attention implementations commonly use F.dropout with a scalar
        # attribute instead of an nn.Dropout child module.
        for name in ("attention_dropout", "hidden_dropout", "dropout"):
            value = getattr(module, name, None)
            if isinstance(value, float) and value:
                setattr(module, name, 0.0)
                count += 1
    return count


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _training_plan(args):
    """Validate inputs before any GPU allocation, including train-only scope."""
    train_file = args.residual_pool_file if args.use_residual_pool else args.train_file
    records = load_jsonl(train_file)
    if not records:
        raise ValueError("The training input is empty")
    ids = []
    for record in records:
        validate_prepared_record(record)
        split = record.get("split", record.get("metadata", {}).get("split"))
        if split != "train":
            raise ValueError(f"RL accepts training records only, found split={split!r}")
        if record.get("metadata", {}).get("task_type", "grounding") != "grounding":
            raise ValueError("The RL reward supports grounding records, not semantic QA")
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("Every training record requires a nonempty string id")
        ids.append(record_id)
        seq_len = _seq_len_for_record(record, _ground_truth_for_record(record))
        if seq_len != args.series_length or args.max_index != seq_len - 1:
            raise ValueError("Record length, --series_length and --max_index must agree")
        if annotation_scope(record) == "localization_only" and args.use_boundary_aware_reward and args.reward_type_weight != 0:
            raise ValueError("Localization-only records cannot receive a type reward")
        if not _resolve_image_path(record, args.image_root).is_file():
            raise FileNotFoundError(f"Image for training id {record_id} is missing")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs in source input; repeats belong only in the explicit prompt schedule")
    validate_generation_budget(records, int(args.max_new_tokens))
    if args.prompt_schedule_file:
        if __package__ in (None, ""):
            from ts_grounder.control_protocol import load_prompt_schedule
        else:
            from .control_protocol import load_prompt_schedule
        if args.max_samples is not None or args.residual_pool_epoch_size is not None:
            raise ValueError("An explicit prompt schedule cannot be combined with truncation or resampling")
        epochs = load_prompt_schedule(args.prompt_schedule_file, records)
    else:
        if args.max_samples is not None:
            records = records[:int(args.max_samples)]
        epoch_size = args.residual_pool_epoch_size or len(records)
        ratios = parse_sampling_ratios(args.residual_pool_sampling_ratios)
        epochs = [
            build_balanced_residual_epoch(records, epoch_size=epoch_size, sampling_ratios=ratios, seed=int(args.seed) + epoch)
            if args.use_residual_pool else random.Random(int(args.seed) + epoch).sample(records, len(records))
            for epoch in range(int(args.num_train_epochs))
        ]
    if not epochs or any(not epoch for epoch in epochs):
        raise ValueError("The prompt schedule must contain nonempty epochs")
    return train_file, records, epochs


def _resume_signature(args, train_file: str | Path) -> dict[str, Any]:
    fields = ("model_name_or_path", "reference_model_path", "num_generations", "max_new_tokens", "temperature",
              "top_p", "learning_rate", "weight_decay", "num_train_epochs", "kl_coef", "use_boundary_aware_reward", "tau_match", "disable_gradient_checkpointing",
              "series_length", "max_index", "prediction_schema", "seed", "optimizer", "torch_dtype",
              "use_residual_pool", "residual_pool_epoch_size", "residual_pool_sampling_ratios", "max_samples",
              "reward_event_f1_weight", "reward_boundary_iou_weight", "reward_hallucination_penalty_weight",
              "reward_point_weight", "reward_event_weight", "reward_iou_weight", "reward_boundary_weight", "reward_type_weight")
    return {
        "implementation": "exact-actions-sequence-mean-v1",
        "grounding_contract_version": GROUNDING_CONTRACT_VERSION,
        "settings": {name: getattr(args, name) for name in fields},
        "train_sha256": _sha256_file(train_file),
        "schedule_sha256": _sha256_file(args.prompt_schedule_file) if args.prompt_schedule_file else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TANGO group-normalized sequence-mean policy optimization (not canonical GRPO).")
    parser.add_argument("--config", type=str, default=None, help="Optional RL config YAML/JSON; CLI args override it.")
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--reference_model_path", type=str, default=None)
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--residual_pool_file", type=str, default=None)
    parser.add_argument("--use_residual_pool", action="store_true")
    parser.add_argument("--prompt_schedule_file", type=str, default=None,
                        help="Exact, fingerprinted epochs/ID schedule; overrides implicit epoch sampling.")
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Explicitly matches the SFT control's zero weight decay by default.")
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--kl_coef", type=float, default=0.02)
    parser.add_argument("--reward_event_f1_weight", type=float, default=DEFAULT_REWARD_WEIGHTS["event_f1"])
    parser.add_argument("--reward_boundary_iou_weight", type=float, default=DEFAULT_REWARD_WEIGHTS["boundary_iou"])
    parser.add_argument(
        "--reward_hallucination_penalty_weight",
        type=float,
        default=DEFAULT_REWARD_WEIGHTS["hallucination_penalty"],
    )
    parser.add_argument("--use_boundary_aware_reward", action="store_true")
    parser.add_argument("--tau_match", type=float, default=DEFAULT_TAU_MATCH)
    parser.add_argument("--tau_good", type=float, default=DEFAULT_TAU_GOOD)
    parser.add_argument("--series_length", type=int, default=DEFAULT_SERIES_LENGTH)
    parser.add_argument("--max_index", type=int, default=DEFAULT_MAX_INDEX)
    parser.add_argument(
        "--prediction_schema",
        type=str,
        default=DEFAULT_PREDICTION_SCHEMA,
        choices=["events", "evidence", "auto"],
    )
    parser.add_argument(
        "--reward_point_weight",
        type=float,
        default=DEFAULT_BOUNDARY_AWARE_REWARD_WEIGHTS["point"],
    )
    parser.add_argument(
        "--reward_event_weight",
        type=float,
        default=DEFAULT_BOUNDARY_AWARE_REWARD_WEIGHTS["event"],
    )
    parser.add_argument(
        "--reward_iou_weight",
        type=float,
        default=DEFAULT_BOUNDARY_AWARE_REWARD_WEIGHTS["iou"],
    )
    parser.add_argument(
        "--reward_boundary_weight",
        type=float,
        default=DEFAULT_BOUNDARY_AWARE_REWARD_WEIGHTS["boundary"],
    )
    parser.add_argument(
        "--reward_type_weight",
        type=float,
        default=DEFAULT_BOUNDARY_AWARE_REWARD_WEIGHTS["type"],
    )
    parser.add_argument(
        "--residual_pool_sampling_ratios",
        type=str,
        default=",".join(f"{key}={value}" for key, value in DEFAULT_RESIDUAL_SAMPLING_RATIOS.items()),
    )
    parser.add_argument("--residual_pool_epoch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--optimizer", type=str, default="adafactor", choices=["adafactor", "adamw"])
    parser.add_argument("--torch_dtype", type=str, default="auto")
    parser.add_argument("--disable_gradient_checkpointing", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--save_optimizer_state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval_file", type=str, default=None)
    parser.add_argument("--eval_output", type=str, default=None)
    parser.add_argument("--eval_max_new_tokens", type=int, default=2048)
    parser.add_argument("--eval_max_samples", type=int, default=None)
    return parser


def _config_defaults_from_file(config_path: str | Path, parser: argparse.ArgumentParser) -> dict[str, Any]:
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    cfg = load_yaml(path)
    if not isinstance(cfg, dict):
        raise ValueError(f"RL config must be an object: {path}")
    source = cfg.get("rl", cfg.get("grpo", cfg))
    if not isinstance(source, dict):
        raise ValueError(f"RL config must contain an object at top-level, rl, or grpo: {path}")

    known_dests = {action.dest for action in parser._actions}
    defaults: dict[str, Any] = {"config": str(path)}
    aliases = {
        "residual_pool_path": "residual_pool_file",
        "residual_pool": "residual_pool_file",
        "error_type_sampling_ratios": "residual_pool_sampling_ratios",
        "sampling_ratios": "residual_pool_sampling_ratios",
    }
    for key, value in source.items():
        dest = aliases.get(str(key), str(key))
        if dest == "reward_weights" and isinstance(value, dict):
            weight_aliases = {
                "event": "reward_event_weight",
                "event_f1": "reward_event_weight",
                "point": "reward_point_weight",
                "point_f1": "reward_point_weight",
                "iou": "reward_iou_weight",
                "mean_iou": "reward_iou_weight",
                "boundary": "reward_boundary_weight",
                "type": "reward_type_weight",
            } if source.get("use_boundary_aware_reward", False) else {
                "event_f1": "reward_event_f1_weight",
                "boundary_iou": "reward_boundary_iou_weight",
                "hallucination_penalty": "reward_hallucination_penalty_weight",
            }
            for weight_key, weight_value in value.items():
                weight_dest = weight_aliases.get(str(weight_key), str(weight_key))
                if weight_dest not in known_dests:
                    raise ValueError(f"Unknown reward setting: {weight_key}")
                defaults[weight_dest] = weight_value
            continue
        if dest not in known_dests:
            raise ValueError(f"Unknown RL setting: {key}")
        defaults[dest] = value
    return defaults


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, remaining = config_parser.parse_known_args()
    if config_args.config:
        parser.set_defaults(**_config_defaults_from_file(config_args.config, parser))
    return parser.parse_args(remaining)


def main() -> None:
    args = parse_args()
    for required_name in ("model_name_or_path", "train_file", "output_dir"):
        if getattr(args, required_name) in (None, ""):
            raise ValueError(f"--{required_name} is required, either on the CLI or in --config")
    if args.use_residual_pool and not args.residual_pool_file:
        raise ValueError("--residual_pool_file is required when --use_residual_pool is enabled")
    if args.num_generations < 2:
        raise ValueError("--num_generations must be at least 2 for group normalization")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("--temperature must be positive when sampling multiple generations")
    if args.top_p != 1.0:
        raise ValueError("Corrected exact-action optimization requires --top_p 1 (and fixed top_k=0); legacy nucleus sampling is not supported")
    if not math.isfinite(args.kl_coef) or args.kl_coef < 0:
        raise ValueError("--kl_coef must be non-negative")
    if args.save_steps < 0:
        raise ValueError("--save_steps must be non-negative")
    if not math.isfinite(args.tau_match) or args.tau_match < 0 or args.tau_match > 1:
        raise ValueError("--tau_match must be in [0, 1]")
    if not math.isfinite(args.tau_good) or args.tau_good < 0 or args.tau_good > 1:
        raise ValueError("--tau_good must be in [0, 1]")
    if args.series_length <= 0:
        raise ValueError("--series_length must be positive")
    if args.max_index < 0:
        raise ValueError("--max_index must be non-negative")
    if args.logging_steps < 1:
        raise ValueError("--logging_steps must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("--learning_rate must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("--weight_decay must be finite and nonnegative")
    if not math.isfinite(args.num_train_epochs) or args.num_train_epochs <= 0 or not float(args.num_train_epochs).is_integer():
        raise ValueError("--num_train_epochs must be a positive integer (fractional epochs are not silently rounded)")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max_samples must be positive")
    if args.residual_pool_epoch_size is not None and args.residual_pool_epoch_size <= 0:
        raise ValueError("--residual_pool_epoch_size must be positive")
    if args.save_steps and not args.save_optimizer_state:
        raise ValueError("Periodic checkpoints require optimizer state for safe resume")
    reward_weights = (
        _boundary_reward_weights_from_args(args)
        if args.use_boundary_aware_reward
        else _reward_weights_from_args(args)
    )
    residual_sampling_ratios = parse_sampling_ratios(args.residual_pool_sampling_ratios)
    if not all(math.isfinite(value) for value in reward_weights.values()):
        raise ValueError("Reward weights must be finite")

    train_file, records, epoch_plan = _training_plan(args)
    signature = _resume_signature(args, train_file)
    if args.prompt_schedule_file:
        if __package__ in (None, ""):
            from ts_grounder.control_protocol import fingerprint_checkpoint
        else:
            from .control_protocol import fingerprint_checkpoint
        schedule_metadata = json.loads(Path(args.prompt_schedule_file).read_text(encoding="utf-8"))
        stage2_fingerprint = fingerprint_checkpoint(args.model_name_or_path)
        if schedule_metadata.get("stage2_checkpoint_fingerprint") != stage2_fingerprint:
            raise ValueError("Prompt schedule belongs to a different Stage-2 checkpoint")
        if args.reference_model_path and fingerprint_checkpoint(args.reference_model_path) != stage2_fingerprint:
            raise ValueError("Controlled runs require the same frozen Stage-2 reference checkpoint")
        signature["stage2_checkpoint_fingerprint"] = stage2_fingerprint
    trainer_state = _load_trainer_state(args.resume_from_checkpoint)
    if args.resume_from_checkpoint:
        if trainer_state.get("run_signature") != signature:
            raise ValueError("Checkpoint settings, source data or prompt schedule differ; exact resume refused")
        if not (Path(args.resume_from_checkpoint) / "optimizer.pt").is_file():
            raise FileNotFoundError("Exact resume requires optimizer.pt")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume_from_checkpoint:
        raise FileExistsError("Refusing to mix a new run with nonempty output_dir; select a fresh output directory")

    eval_records = None
    if args.eval_file:
        eval_records = load_jsonl(args.eval_file)
        if args.eval_max_samples is not None:
            if args.eval_max_samples <= 0:
                raise ValueError("--eval_max_samples must be positive")
            eval_records = eval_records[: int(args.eval_max_samples)]
        train_ids = {record["id"] for record in records}
        for record in eval_records:
            validate_prepared_record(record)
            if record.get("split") not in {"val", "test"} or record["id"] in train_ids:
                raise ValueError("Post-training evaluation requires disjoint val/test records")
        validate_generation_budget(eval_records, int(args.eval_max_new_tokens))

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    output_dir.mkdir(parents=True, exist_ok=True)
    rollout_path = output_dir / "rl_rollouts.jsonl"
    if args.resume_from_checkpoint and rollout_path.exists():
        with rollout_path.open(encoding="utf-8") as stream:
            final_logged_step = 0
            for line in stream:
                if line.strip():
                    final_logged_step = int(json.loads(line).get("step", -1))
        if final_logged_step != int(trainer_state.get("global_step", 0)):
            raise ValueError("Existing rollout log is not at the checkpoint boundary; resume into a fresh output_dir to preserve history")

    _seed_step(int(args.seed), 0)
    model_load_path = args.resume_from_checkpoint or args.model_name_or_path
    processor = AutoProcessor.from_pretrained(model_load_path, trust_remote_code=True)
    model_kwargs: dict[str, Any] = {}
    torch_dtype = _resolve_dtype(args.torch_dtype)
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    model = AutoModelForImageTextToText.from_pretrained(
        model_load_path,
        trust_remote_code=True,
        **model_kwargs,
    )
    if getattr(model.config, "is_encoder_decoder", False):
        raise ValueError("This action alignment implementation supports decoder-only VLMs")
    dropout_modules_disabled = _disable_dropout(model)
    gradient_checkpointing = not bool(args.disable_gradient_checkpointing)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = _build_optimizer(model, args)
    optimizer_resumed = _maybe_resume_optimizer(optimizer, args.resume_from_checkpoint)

    reference_model = None
    reference_model_path = args.reference_model_path or args.model_name_or_path
    if args.kl_coef > 0:
        reference_model = AutoModelForImageTextToText.from_pretrained(
            reference_model_path,
            trust_remote_code=True,
            **model_kwargs,
        )
        if hasattr(reference_model.config, "use_cache"):
            reference_model.config.use_cache = False
        reference_model = reference_model.to(device)
        reference_model.eval()
        for parameter in reference_model.parameters():
            parameter.requires_grad_(False)

    residual_pool_epoch_size = (
        int(args.residual_pool_epoch_size)
        if args.residual_pool_epoch_size is not None
        else len(records)
    )
    num_epochs = len(epoch_plan)
    global_step = int(trainer_state.get("global_step", 0))
    start_epoch = int(trainer_state.get("next_epoch", 0))
    start_record_index = int(trainer_state.get("next_record_index", 0))
    if start_epoch < 0 or start_epoch > num_epochs or start_record_index < 0:
        raise ValueError("Invalid checkpoint schedule position")
    if start_epoch < num_epochs and start_record_index >= len(epoch_plan[start_epoch]):
        raise ValueError("Checkpoint points beyond its epoch")
    expected_step = sum(len(epoch) for epoch in epoch_plan[:start_epoch]) + start_record_index
    if global_step != expected_step or (start_epoch == num_epochs and start_record_index != 0):
        raise ValueError("Checkpoint global_step disagrees with the exact prompt schedule")
    running_rewards: list[float] = []

    print(
        "RL config:",
        {
            "kl_coef": float(args.kl_coef),
            "config": args.config,
            "use_boundary_aware_reward": bool(args.use_boundary_aware_reward),
            "reward_weights": reward_weights,
            "tau_match": float(args.tau_match),
            "tau_good": float(args.tau_good),
            "series_length": int(args.series_length),
            "max_index": int(args.max_index),
            "prediction_schema": str(args.prediction_schema),
            "use_residual_pool": bool(args.use_residual_pool),
            "residual_pool_file": args.residual_pool_file,
            "residual_pool_epoch_size": residual_pool_epoch_size if args.use_residual_pool else None,
            "residual_pool_sampling_ratios": residual_sampling_ratios if args.use_residual_pool else None,
            "reference_model_path": reference_model_path if reference_model is not None else None,
            "optimizer": str(args.optimizer),
            "gradient_checkpointing": gradient_checkpointing,
            "save_steps": int(args.save_steps),
            "resume_from_checkpoint": args.resume_from_checkpoint,
            "optimizer_resumed": optimizer_resumed,
            "prompt_schedule_file": args.prompt_schedule_file,
            "objective": "group-normalized sequence-mean policy gradient plus squared reference penalty",
            "dropout_modules_disabled": dropout_modules_disabled,
        },
    )

    for epoch in range(start_epoch, num_epochs):
        epoch_records = epoch_plan[epoch]
        epoch_start_index = start_record_index if epoch == start_epoch else 0
        for record_index in range(epoch_start_index, len(epoch_records)):
            record = epoch_records[record_index]
            image_path = _resolve_image_path(record, args.image_root)
            with Image.open(image_path) as opened_image:
                image = opened_image.convert("RGB")
            system_prompt = str(record["system_prompt"])
            user_prompt = str(record["user_prompt"])
            ground_truth = _ground_truth_for_record(record)
            seq_len = _seq_len_for_record(record, ground_truth)

            rollout_seed = _seed_step(int(args.seed), global_step)
            rollout = _generate_candidates(
                processor=processor,
                model=model,
                image=image,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                device=device,
                num_generations=int(args.num_generations),
                max_new_tokens=int(args.max_new_tokens),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
            )
            image.close()
            candidates = rollout["texts"]
            if args.use_boundary_aware_reward:
                reward_details = [
                    compute_boundary_aware_grounder_reward_details(
                        candidate,
                        ground_truth,
                        seq_len,
                        tau_match=float(args.tau_match),
                        max_index=int(args.max_index),
                        reward_weights=reward_weights,
                        prediction_schema=str(args.prediction_schema),
                        annotation_scope=annotation_scope(record),
                    )
                    for candidate in candidates
                ]
            else:
                reward_details = [
                    compute_grounder_reward_details(
                        candidate,
                        ground_truth,
                        seq_len,
                        reward_weights=reward_weights,
                        annotation_scope=annotation_scope(record),
                    )
                    for candidate in candidates
                ]
            rewards = [float(item["reward"]) for item in reward_details]
            advantages = _normalize_advantages(rewards)

            ref_log_probs = None
            if reference_model is not None:
                with torch.no_grad():
                    ref_log_probs = _candidate_log_probs(
                        model=reference_model,
                        rollout=rollout,
                        temperature=float(args.temperature),
                    )

            model.train()
            log_probs = _candidate_log_probs(
                model=model,
                rollout=rollout,
                temperature=float(args.temperature),
            )
            loss, policy_loss, kl_penalty, sampled_kl = _policy_objective(
                log_probs, advantages, ref_log_probs, reference_coef=float(args.kl_coef)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradients = [param.grad for param in model.parameters() if param.grad is not None]
            if not gradients or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError("Missing or non-finite policy gradients; optimizer step aborted")
            optimizer.step()

            global_step += 1
            running_rewards.extend(rewards)
            next_epoch, next_record_index = _next_position(epoch, record_index, len(epoch_records))
            _append_jsonl(
                rollout_path,
                {
                    "epoch": epoch,
                    "record_index": record_index,
                    "sample_id": record.get("id") or record.get("sample_id"),
                    "step": global_step,
                    "rollout_seed": rollout_seed,
                    "sampled_token_ids": rollout["completion_ids"].detach().cpu().tolist(),
                    "response_lengths": rollout["completion_mask"].sum(dim=1).detach().cpu().tolist(),
                    "candidates": candidates,
                    "rewards": rewards,
                    "advantages": advantages,
                    "reward_details": reward_details,
                    "loss": float(loss.detach().cpu()),
                    "policy_loss": float(policy_loss.detach().cpu()),
                    "kl_penalty": float(kl_penalty.detach().cpu()),
                    "sampled_kl": float(sampled_kl.detach().cpu()),
                    "ref_log_probs": (
                        None
                        if ref_log_probs is None
                        else [float(value) for value in ref_log_probs.detach().cpu().tolist()]
                    ),
                    "next_epoch": next_epoch,
                    "next_record_index": next_record_index,
                },
            )

            if global_step % int(args.logging_steps) == 0:
                recent = running_rewards[-int(args.logging_steps) * int(args.num_generations) :]
                mean_reward = sum(recent) / len(recent) if recent else 0.0
                print(
                    f"step={global_step} epoch={epoch} "
                    f"mean_recent_reward={mean_reward:.4f} "
                    f"loss={float(loss.detach().cpu()):.4f} "
                    f"kl_penalty={float(kl_penalty.detach().cpu()):.6f}"
                )
            if args.save_steps and global_step % int(args.save_steps) == 0:
                _save_checkpoint(
                    checkpoint_dir=output_dir / "checkpoints" / f"step-{global_step}",
                    model=model,
                    processor=processor,
                    optimizer=optimizer,
                    state={
                        "global_step": global_step,
                        "next_epoch": next_epoch,
                        "next_record_index": next_record_index,
                        "num_train_epochs": float(args.num_train_epochs),
                        "train_file": train_file,
                        "use_residual_pool": bool(args.use_residual_pool),
                        "use_boundary_aware_reward": bool(args.use_boundary_aware_reward),
                        "model_name_or_path": args.model_name_or_path,
                        "reference_model_path": reference_model_path,
                        "kl_coef": float(args.kl_coef),
                        "reward_weights": reward_weights,
                        "tau_match": float(args.tau_match),
                        "tau_good": float(args.tau_good),
                        "series_length": int(args.series_length),
                        "max_index": int(args.max_index),
                        "prediction_schema": str(args.prediction_schema),
                        "run_signature": signature,
                    },
                    save_optimizer_state=bool(args.save_optimizer_state),
                )

    model_dir = output_dir / "model"
    _save_checkpoint(
        checkpoint_dir=model_dir, model=model, processor=processor, optimizer=optimizer,
        state={"global_step": global_step, "next_epoch": num_epochs, "next_record_index": 0,
               "run_signature": signature}, save_optimizer_state=bool(args.save_optimizer_state),
    )

    eval_summary = None
    if eval_records is not None:
        eval_output = Path(args.eval_output) if args.eval_output else output_dir / "eval" / "rl_predictions.jsonl"
        eval_device = _model_device(model)
        eval_summary = _generate_predictions_for_records(
            processor=processor,
            model=model,
            records=eval_records,
            output_path=eval_output,
            max_new_tokens=int(args.eval_max_new_tokens),
            device=eval_device,
        )
        eval_metrics_path = eval_output.with_suffix(".metrics.json")
        eval_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        eval_metrics_path.write_text(
            json.dumps(eval_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    summary = {
        "mode": "grounder_group_normalized_sequence_mean_policy_optimization",
        "model_name_or_path": args.model_name_or_path,
        "reference_model_path": reference_model_path if reference_model is not None else None,
        "train_file": train_file,
        "image_root": args.image_root,
        "output_dir": str(output_dir),
        "num_generations": int(args.num_generations),
        "max_new_tokens": int(args.max_new_tokens),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "num_train_epochs": float(args.num_train_epochs),
        "scheduled_epochs": len(epoch_plan),
        "prompt_schedule_file": args.prompt_schedule_file,
        "run_signature": signature,
        "kl_coef": float(args.kl_coef),
        "config": args.config,
        "use_boundary_aware_reward": bool(args.use_boundary_aware_reward),
        "reward_weights": reward_weights,
        "tau_match": float(args.tau_match),
        "tau_good": float(args.tau_good),
        "series_length": int(args.series_length),
        "max_index": int(args.max_index),
        "prediction_schema": str(args.prediction_schema),
        "use_residual_pool": bool(args.use_residual_pool),
        "residual_pool_file": args.residual_pool_file,
        "residual_pool_epoch_size": residual_pool_epoch_size if args.use_residual_pool else None,
        "residual_pool_sampling_ratios": residual_sampling_ratios if args.use_residual_pool else None,
        "optimizer": str(args.optimizer),
        "gradient_checkpointing": gradient_checkpointing,
        "kl_reference_active": reference_model is not None,
        "reference_regularizer": "0.5 * mean((sequence_mean_logp_policy - sequence_mean_logp_reference)^2); NOT KL",
        "likelihood_temperature": float(args.temperature),
        "top_k": 0,
        "dropout_modules_disabled": dropout_modules_disabled,
        "save_steps": int(args.save_steps),
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "optimizer_resumed": optimizer_resumed,
        "save_optimizer_state": bool(args.save_optimizer_state),
        "num_steps": global_step,
        "mean_reward_this_process": float(sum(running_rewards) / len(running_rewards)) if running_rewards else None,
        "rollouts_path": str(rollout_path),
        "model_dir": str(model_dir),
        "eval": eval_summary,
    }
    (output_dir / "rl_train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
