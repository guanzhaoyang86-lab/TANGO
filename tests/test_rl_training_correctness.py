"""Real CPU-autograd regressions; no model download or GPU is required."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
torch = pytest.importorskip("torch")

from ts_grounder.rl_train_grpo import (
    _candidate_log_probs,
    _build_optimizer,
    _config_defaults_from_file,
    _completion_mask,
    _disable_dropout,
    _generate_candidates,
    _ground_truth_for_record,
    _maybe_resume_optimizer,
    _normalize_advantages,
    _policy_objective,
    _seed_step,
    _training_plan,
    build_parser,
)
from ts_grounder.rl_reward import compute_boundary_aware_grounder_reward_details, compute_grounder_reward_details
from ts_grounder.supervision import TARGET_ADAPTER_VERSION, LOCALIZATION_SYSTEM_PROMPT


def test_eos_mask_includes_first_eos_even_when_pad_equals_eos():
    ids = torch.tensor([[5, 2, 2, 2], [2, 2, 2, 2], [5, 6, 7, 8]])
    assert _completion_mask(ids, 2).tolist() == [[True, True, False, False], [True, False, False, False], [True] * 4]


def test_eos_mask_multiple_stop_tokens_and_pad_inside_action():
    ids = torch.tensor([[0, 5, 3, 0], [5, 2, 0, 0]])
    assert _completion_mask(ids, [2, 3]).tolist() == [[True, True, True, False], [True, True, False, False]]


class TinyPolicy(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = torch.nn.Parameter(logits.clone())

    def forward(self, **batch):
        return SimpleNamespace(logits=self.logits)


def make_rollout():
    ids = torch.tensor([[0, 1, 2, 3, 3], [0, 1, 2, 2, 3]])
    mask = _completion_mask(ids[:, 2:], 3)
    return {"model_inputs": {"input_ids": ids}, "prompt_length": 2, "completion_mask": mask}


def test_exact_action_logprob_temperature_eos_and_prompt_mask():
    rollout = make_rollout()
    torch.manual_seed(3)
    model = TinyPolicy(torch.randn(2, 5, 4, dtype=torch.float64))
    values = _candidate_log_probs(model=model, rollout=rollout, temperature=0.7)
    token_values = torch.log_softmax(model.logits.float()[:, 1:4, :] / 0.7, dim=-1)
    selected = token_values.gather(-1, rollout["model_inputs"]["input_ids"][:, 2:].unsqueeze(-1)).squeeze(-1)
    expected = torch.stack([selected[0, :2].mean(), selected[1].mean()])
    assert torch.allclose(values, expected)
    values.sum().backward()
    assert torch.count_nonzero(model.logits.grad[:, 0, :]) == 0  # prompt prediction is never trained
    assert torch.count_nonzero(model.logits.grad[:, 4, :]) == 0  # no next token at final position
    assert torch.count_nonzero(model.logits.grad[0, 3, :]) == 0  # padding after EOS
    assert torch.count_nonzero(model.logits.grad[0, 2, :]) > 0  # first EOS is trained


def test_vision_action_mask_is_renormalized_but_prompt_image_remains_valid():
    logits = torch.zeros(1, 4, 4)
    logits[..., 2] = 10.0
    model = TinyPolicy(logits)
    model.config = SimpleNamespace(image_token_id=2)
    # The image token in the input prompt is legal. It cannot be generated in
    # the JSON response, which contains only a text token and EOS.
    ids = torch.tensor([[0, 2, 1, 3]])
    rollout = {"prompt_length": 2, "completion_mask": torch.ones(1, 2, dtype=torch.bool),
               "model_inputs": {"input_ids": ids}, "forbidden_action_token_ids": [2]}
    values = _candidate_log_probs(model=model, rollout=rollout)
    assert values.item() == pytest.approx(-math.log(3))
    values.sum().backward()
    assert torch.count_nonzero(model.logits.grad[..., 2]) == 0
    rollout["model_inputs"]["input_ids"] = torch.tensor([[0, 2, 2, 3]])
    with pytest.raises(ValueError, match="forbidden vision"):
        _candidate_log_probs(model=model, rollout=rollout)


def test_optimizers_explicitly_match_sft_zero_weight_decay():
    model = TinyPolicy(torch.zeros(1, 3, 4))
    args = build_parser().parse_args([])
    for name in ("adafactor", "adamw"):
        args.optimizer = name
        optimizer = _build_optimizer(model, args)
        assert all(group["weight_decay"] == 0 for group in optimizer.param_groups)


def test_group_advantages_and_original_objective_have_correct_gradient():
    policy = torch.tensor([-0.8, -1.2], requires_grad=True)
    reference = torch.tensor([-1.0, -1.0], requires_grad=True)
    advantages = torch.tensor(_normalize_advantages([1.0, 0.0]), requires_grad=True)
    loss, pg, penalty, ratio = _policy_objective(policy, advantages, reference, reference_coef=0.03)
    expected = -(advantages.detach() * policy).mean() + 0.03 * 0.5 * ((policy - reference.detach()) ** 2).mean()
    assert torch.allclose(loss, expected)
    loss.backward()
    assert policy.grad[0] < 0 and policy.grad[1] > 0
    assert reference.grad is None and advantages.grad is None
    assert penalty.item() == pytest.approx(0.02)


def test_zero_variance_rewards_are_finite_zero_and_nonfinite_rewards_rejected():
    assert _normalize_advantages([0.5, 0.5]) == [0.0, 0.0]
    with pytest.raises(ValueError, match="finite"):
        _normalize_advantages([math.nan, 1.0])
    with pytest.raises(FloatingPointError):
        _policy_objective(torch.tensor([float("inf")]), [1])


def test_per_step_rng_reproduces_after_unrelated_draws():
    _seed_step(17, 5)
    expected = torch.randn(16)
    torch.randn(100)
    _seed_step(17, 5)
    assert torch.equal(torch.randn(16), expected)
    _seed_step(17, 6)
    assert not torch.equal(torch.randn(16), expected)


def test_optimizer_resume_missing_state_is_not_silently_accepted(tmp_path):
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    with pytest.raises(FileNotFoundError, match="optimizer.pt"):
        _maybe_resume_optimizer(optimizer, tmp_path)
    (parameter.square().sum()).backward()
    optimizer.step()
    torch.save(optimizer.state_dict(), tmp_path / "optimizer.pt")
    resumed = torch.optim.AdamW([torch.nn.Parameter(torch.tensor([1.0]))], lr=0.01)
    assert _maybe_resume_optimizer(resumed, tmp_path)
    assert resumed.state_dict()["state"][0]["step"].item() == 1


def test_dropout_disabled_in_train_mode():
    model = torch.nn.Sequential(torch.nn.Dropout(0.5))
    model.attention_dropout = 0.3
    assert _disable_dropout(model) == 2
    model.train()
    assert torch.equal(model(torch.ones(20)), torch.ones(20))
    assert model.attention_dropout == 0


class FakeProcessor:
    tokenizer = SimpleNamespace(pad_token_id=3, eos_token_id=3)

    def apply_chat_template(self, messages, **kwargs):
        assert "assistant_text" not in kwargs
        return "unchanged prompt"

    def __call__(self, text, images, **kwargs):
        return {"input_ids": torch.tensor([[0, 1]] * len(text)),
                "attention_mask": torch.ones(len(text), 2, dtype=torch.long),
                "pixel_values": torch.ones(len(text), 3),
                "mm_token_type_ids": torch.tensor([[1, 0]] * len(text))}

    def batch_decode(self, token_ids, **kwargs):
        return ["decode does not roundtrip"] * token_ids.shape[0]


class FakeGenerator(TinyPolicy):
    generation_config = SimpleNamespace(eos_token_id=[3])

    def generate(self, **kwargs):
        assert kwargs["top_k"] == 0 and kwargs["top_p"] == 1.0
        assert kwargs["temperature"] == 0.7 and kwargs["num_beams"] == 1
        return torch.tensor([[0, 1, 2, 3, 3], [0, 1, 2, 2, 3]])


def test_rollout_preserves_exact_tokens_without_decode_reencode():
    model = FakeGenerator(torch.zeros(2, 5, 4))
    rollout = _generate_candidates(processor=FakeProcessor(), model=model, image=object(), system_prompt="s",
                                   user_prompt="u", device="cpu", num_generations=2, max_new_tokens=3,
                                   temperature=0.7, top_p=1.0)
    assert rollout["completion_ids"].tolist() == [[2, 3, 3], [2, 2, 3]]
    assert rollout["model_inputs"]["attention_mask"].tolist() == [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]
    assert rollout["model_inputs"]["mm_token_type_ids"].shape == (2, 5)
    _candidate_log_probs(model=model, rollout=rollout).sum().backward()
    assert model.logits.grad is not None


def test_localization_reward_has_no_semantic_requirements_or_type_reward():
    payload = {"evidence": [{"start": 2, "end": 4}]}
    details = compute_boundary_aware_grounder_reward_details(
        json.dumps(payload), payload, 8, max_index=7, annotation_scope="localization_only",
        reward_weights={"point": 0.45, "event": 0, "iou": 0.4, "boundary": 0.15, "type": 0})
    assert details["reward"] == pytest.approx(1)
    assert details["type_score"] is None
    extra = {**payload, "summary": "invented"}
    rejected = compute_boundary_aware_grounder_reward_details(
        json.dumps(extra), payload, 8, max_index=7, annotation_scope="localization_only")
    assert rejected["reward"] == 0
    with pytest.raises(ValueError, match="type reward"):
        compute_boundary_aware_grounder_reward_details(payload, payload, 8, max_index=7,
            annotation_scope="localization_only", reward_weights={"type": 1})


def test_legacy_reward_wrapper_also_accepts_strict_localization():
    payload = {"evidence": [{"start": 2, "end": 4}]}
    assert compute_grounder_reward_details(payload, payload, 8, annotation_scope="localization_only")["schema_valid"] == 1
    assert compute_grounder_reward_details({**payload, "summary": "wrong"}, payload, 8,
                                           annotation_scope="localization_only")["reward"] == -1


def test_conflicting_rl_target_is_rejected():
    with pytest.raises(ValueError, match="disagree"):
        _ground_truth_for_record({"target": {"evidence": []}, "ground_truth": {"evidence": [{"start": 1, "end": 2}]}})


def test_config_reward_weights_map_to_selected_reward_and_reject_typos(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"rl": {"reward_weights": {"event_f1": 0.2}}}), encoding="utf-8")
    assert _config_defaults_from_file(path, build_parser())["reward_event_f1_weight"] == 0.2
    path.write_text(json.dumps({"rl": {"use_boundary_aware_reward": True, "reward_weights": {"point": 0.45}}}), encoding="utf-8")
    assert _config_defaults_from_file(path, build_parser())["reward_point_weight"] == 0.45
    path.write_text(json.dumps({"rl": {"learning_rate_typo": 0.1}}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown RL setting"):
        _config_defaults_from_file(path, build_parser())


def test_nontrain_or_old_cache_rejected_before_model_load(tmp_path):
    args = build_parser().parse_args([])
    args.train_file = str(tmp_path / "train.jsonl")
    record = {"id": "r1", "split": "test", "metadata": {"target_adapter_version": "old"}}
    Path(args.train_file).write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="Old/unversioned"):
        _training_plan(args)
    payload = {"evidence": []}
    record.update(target=payload, assistant_text=json.dumps(payload), system_prompt=LOCALIZATION_SYSTEM_PROMPT)
    record["metadata"].update(target_adapter_version=TARGET_ADAPTER_VERSION, annotation_scope="localization_only",
        indexed_series_text_enabled=True, indexed_series_precision=6, series_length=256)
    Path(args.train_file).write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="training records only"):
        _training_plan(args)


def test_real_torch_two_step_training_and_checkpoint_resume_match(tmp_path, monkeypatch):
    """Exercise main, sampled-ID scoring, optimizer updates and exact resume.

    The tiny CPU module and processor are local fixtures, not a Qwen integration
    test; both gradients and optimizer state are genuine torch operations.
    """
    import transformers
    from PIL import Image
    from ts_grounder import rl_train_grpo

    class TrainProcessor(FakeProcessor):
        def batch_decode(self, token_ids, **kwargs):
            return ['{"evidence":[{"start":2,"end":4}]}', '{"evidence":[]}']

        def save_pretrained(self, folder):
            Path(folder).mkdir(parents=True, exist_ok=True)

    class TrainPolicy(FakeGenerator):
        config = SimpleNamespace(use_cache=False, is_encoder_decoder=False)

        def gradient_checkpointing_enable(self):
            pass

        def save_pretrained(self, folder):
            Path(folder).mkdir(parents=True, exist_ok=True)
            torch.save(self.state_dict(), Path(folder) / "test_weights.pt")

    def load_policy(folder, **kwargs):
        model = TrainPolicy(torch.zeros(2, 5, 4))
        weights = Path(folder) / "test_weights.pt"
        if weights.is_file():
            model.load_state_dict(torch.load(weights, weights_only=True))
        return model

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *a, **k: TrainProcessor())
    monkeypatch.setattr(transformers.AutoModelForImageTextToText, "from_pretrained", load_policy)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 8)).save(image)
    payload = {"evidence": [{"start": 2, "end": 4}]}
    metadata = dict(target_adapter_version=TARGET_ADAPTER_VERSION, annotation_scope="localization_only",
                    indexed_series_text_enabled=True, indexed_series_precision=6, series_length=256)
    records = [{"id": f"r{i}", "split": "train", "image_path": str(image), "target": payload,
                "assistant_text": json.dumps(payload), "system_prompt": LOCALIZATION_SYSTEM_PROMPT,
                "user_prompt": "Indexed values: 0: 0.000000", "metadata": metadata} for i in range(2)]
    train_file = tmp_path / "train.jsonl"
    train_file.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    common = ["rl_train_grpo", "--model_name_or_path", str(tmp_path / "stage2"), "--train_file", str(train_file),
              "--num_generations", "2", "--optimizer", "adamw", "--learning_rate", "0.01", "--save_steps", "1",
              "--use_boundary_aware_reward", "--torch_dtype", "float32"]
    full = tmp_path / "full"
    monkeypatch.setattr(sys, "argv", common + ["--output_dir", str(full)])
    rl_train_grpo.main()
    full_weights = torch.load(full / "model" / "test_weights.pt", weights_only=True)["logits"]
    assert torch.count_nonzero(full_weights) > 0
    checkpoint = full / "checkpoints" / "step-1"
    resumed = tmp_path / "resumed"
    monkeypatch.setattr(sys, "argv", common + ["--output_dir", str(resumed), "--resume_from_checkpoint", str(checkpoint)])
    rl_train_grpo.main()
    resumed_weights = torch.load(resumed / "model" / "test_weights.pt", weights_only=True)["logits"]
    assert torch.equal(full_weights, resumed_weights)
    resumed_state = json.loads((resumed / "model" / "trainer_state.json").read_text())
    assert resumed_state["global_step"] == 2
    assert resumed_state["next_epoch"] == 1


def test_tiny_real_qwen25_vl_packed_visual_forward_and_autograd():
    """Real Transformers architecture, tiny random weights, no network access."""
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
    config = Qwen2_5_VLConfig(
        text_config=dict(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                         num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128,
                         rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]},
                         pad_token_id=0, eos_token_id=2, bos_token_id=1),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4, in_channels=3,
                           patch_size=2, temporal_patch_size=1, spatial_merge_size=2, window_size=4,
                           out_hidden_size=32, fullatt_block_indexes=[0]),
        image_token_id=21, video_token_id=23, vision_start_token_id=20, vision_end_token_id=22,
    )
    config._attn_implementation = "eager"
    model = Qwen2_5_VLForConditionalGeneration(config)
    model.train()
    ids = torch.tensor([[1, 20, 21, 22, 4, 7, 2, 2], [1, 20, 21, 22, 4, 6, 8, 2]])
    mask = _completion_mask(ids[:, 5:], 2)
    rollout = {"prompt_length": 5, "completion_mask": mask,
               "model_inputs": {"input_ids": ids, "attention_mask": torch.cat((torch.ones(2, 5, dtype=torch.long), mask.long()), 1),
                                "pixel_values": torch.randn(8, 12), "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]])}}
    log_probs = _candidate_log_probs(model=model, rollout=rollout, temperature=0.7)
    assert log_probs.shape == (2,) and torch.isfinite(log_probs).all()
    loss, *_ = _policy_objective(log_probs, [1.0, -1.0])
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) for gradient in gradients)
