"""Offline CPU integration with the real Transformers Qwen2.5-VL architecture.

All weights/tokenizer vocabulary are initialized locally and are deliberately
tiny. This verifies tensor plumbing/autograd, NOT pretrained-model accuracy,
GPU capacity, convergence, or the reported research results.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _tiny_qwen_components():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (Qwen2TokenizerFast, Qwen2VLImageProcessor,
                              Qwen2VLVideoProcessor, Qwen2_5_VLConfig,
                              Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor)

    tokens = ["<unk>", "<pad>", "<|im_start|>", "<|im_end|>",
              "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>",
              "system", "user", "assistant", "evidence", "start", "end",
              "{", "}", "[", "]", ":", ",", '"', "10", "12"]
    backend = Tokenizer(models.WordLevel({word: i for i, word in enumerate(tokens)}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = Qwen2TokenizerFast(tokenizer_object=backend, unk_token="<unk>",
        pad_token="<pad>", eos_token="<|im_end|>",
        additional_special_tokens=tokens[2:8], padding_side="right")
    template = (
        "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' }}"
        "{% for item in message['content'] %}"
        "{% if item['type'] == 'image' %}{{ '<|vision_start|><|image_pad|><|vision_end|>' }}"
        "{% elif item['type'] == 'text' %}{{ item['text'] }}{% endif %}{% endfor %}"
        "{{ '<|im_end|>\n' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    )
    image_processor = Qwen2VLImageProcessor(min_pixels=56 * 56, max_pixels=56 * 56,
                                            patch_size=14, temporal_patch_size=2, merge_size=2)
    processor = Qwen2_5_VLProcessor(image_processor=image_processor, tokenizer=tokenizer,
                                  video_processor=Qwen2VLVideoProcessor(), chat_template=template)
    config = Qwen2_5_VLConfig(
        text_config={"vocab_size": len(tokenizer), "hidden_size": 32, "intermediate_size": 64,
                     "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2,
                     "max_position_embeddings": 4096, "use_cache": False,
                     "rope_scaling": {"type": "default", "mrope_section": [1, 1, 2]},
                     "bos_token_id": tokenizer.convert_tokens_to_ids("<|im_start|>"),
                     "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id},
        vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64,
                       "out_hidden_size": 32, "num_heads": 4, "patch_size": 14,
                       "spatial_merge_size": 2, "temporal_patch_size": 2,
                       "fullatt_block_indexes": [0], "window_size": 112},
        image_token_id=tokenizer.convert_tokens_to_ids("<|image_pad|>"),
        video_token_id=tokenizer.convert_tokens_to_ids("<|video_pad|>"),
        vision_start_token_id=tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        vision_end_token_id=tokenizer.convert_tokens_to_ids("<|vision_end|>"),
    )
    model = Qwen2_5_VLForConditionalGeneration(config)
    return processor, model


def test_real_tiny_qwen_vlm_processor_collator_forward_gradient_and_save():
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from ts_grounder.vlm_data import _record_from_sample
    from ts_grounder.vlm_training import VisionChatCollator
    from ts_grounder.control_protocol import fingerprint_checkpoint

    torch.manual_seed(2026)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        processor, model = _tiny_qwen_components()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (56, 56), (40, 100, 170)).save(root / "sample.png")
            records = []
            for anomalous in (False, True):
                sample = {"sample_id": str(anomalous), "series": [float(i) / 255 for i in range(256)],
                          "point_labels": [int(anomalous and 10 <= i <= 12) for i in range(256)],
                          "events": [{"start": 10, "end": 12}] if anomalous else [],
                          "image_path": "sample.png", "annotation_scope": "localization_only",
                          "semantic_label_available": False, "context": {"series_length": 256}}
                records.append(_record_from_sample(sample, source_root=root, output_root=root / "cache",
                    split="train", system_prompt="ignored", user_prompt="ignored"))
            batch = VisionChatCollator(processor)(records)
            assert "pixel_values" in batch and "image_grid_thw" in batch
            assert (batch["labels"] != -100).any(dim=1).all()
            assert (batch["labels"][batch["input_ids"] == model.config.image_token_id] == -100).all()
            assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            before = model.get_input_embeddings().weight.detach().clone()
            result = model(**batch)
            assert result.loss.ndim == 0 and torch.isfinite(result.loss)
            result.loss.backward()
            vision_gradients = [parameter.grad for name, parameter in model.named_parameters()
                                if "visual" in name and parameter.grad is not None]
            assert vision_gradients and any(gradient.abs().sum() > 0 for gradient in vision_gradients)
            assert all(torch.isfinite(gradient).all() for gradient in vision_gradients)
            optimizer.step()
            assert not torch.equal(before, model.get_input_embeddings().weight)

            checkpoint = root / "checkpoint"
            model.save_pretrained(checkpoint)
            processor.save_pretrained(checkpoint)
            assert len(fingerprint_checkpoint(checkpoint)) == 64
            restored_processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
            restored_model = AutoModelForImageTextToText.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
            restored_batch = VisionChatCollator(restored_processor)(records)
            with torch.no_grad():
                assert torch.isfinite(restored_model(**restored_batch).loss)
    finally:
        torch.set_num_threads(old_threads)


def test_real_tiny_qwen_rollout_expands_images_and_preserves_action_gradient():
    import torch
    from PIL import Image
    from ts_grounder.rl_train_grpo import _candidate_log_probs, _generate_candidates, _policy_objective

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    try:
        processor, model = _tiny_qwen_components()
        rollout = _generate_candidates(processor=processor, model=model,
            image=Image.new("RGB", (56, 56), (40, 100, 170)), system_prompt="Locate anomalies.",
            user_prompt="Indexed values: 0:0, 1:1, 2:5.", device=torch.device("cpu"),
            num_generations=2, max_new_tokens=3, temperature=0.7, top_p=1.0)
        assert len(rollout["texts"]) == 2
        assert rollout["model_inputs"]["image_grid_thw"].shape[0] == 2
        scores = _candidate_log_probs(model=model, rollout=rollout, temperature=0.7)
        assert scores.shape == (2,) and torch.isfinite(scores).all()
        loss, _, _, _ = _policy_objective(scores, [1.0, -1.0])
        loss.backward()
        assert model.get_input_embeddings().weight.grad is not None
        assert torch.isfinite(model.get_input_embeddings().weight.grad).all()
    finally:
        torch.set_num_threads(old_threads)
