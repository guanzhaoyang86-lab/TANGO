"""Offline regression using real HF save/load, not a processor test double."""
from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_actual_qwen_processor_template_changes_change_checkpoint_identity(tmp_path):
    import torch
    from transformers import AutoProcessor
    from transformers.utils import CHAT_TEMPLATE_DIR
    from test_tiny_vlm_integration import _tiny_qwen_components
    from ts_grounder.control_protocol import fingerprint_checkpoint

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        processor, model = _tiny_qwen_components()
        checkpoint = tmp_path / "checkpoint"
        model.save_pretrained(checkpoint)
        processor.save_pretrained(checkpoint)
        default_file = checkpoint / "chat_template.jinja"
        assert default_file.is_file()  # Exercise the actual HF serialization.
        messages = [{"role": "user", "content": [{"type": "text", "text": "Locate anomalies."}]}]

        def rendered_prompt(template_name=None):
            loaded = AutoProcessor.from_pretrained(
                checkpoint, local_files_only=True, trust_remote_code=False)
            kwargs = {"chat_template": template_name} if template_name else {}
            return loaded.apply_chat_template(messages, tokenize=False, **kwargs)

        original_prompt = rendered_prompt()
        baseline = fingerprint_checkpoint(checkpoint)
        default_file.write_text(default_file.read_text(encoding="utf-8") + " changed-default", encoding="utf-8")
        assert rendered_prompt() != original_prompt
        assert fingerprint_checkpoint(checkpoint) != baseline

        # Transformers persists a default and named templates in different paths.
        processor.chat_template = {"default": processor.chat_template, "tool_use": processor.chat_template + " tools"}
        processor.save_pretrained(checkpoint)
        named_file = checkpoint / CHAT_TEMPLATE_DIR / "tool_use.jinja"
        assert named_file.is_file()
        original_named_prompt = rendered_prompt("tool_use")
        named_baseline = fingerprint_checkpoint(checkpoint)
        named_file.write_text(named_file.read_text(encoding="utf-8") + " changed-named", encoding="utf-8")
        assert rendered_prompt("tool_use") != original_named_prompt
        assert fingerprint_checkpoint(checkpoint) != named_baseline
    finally:
        torch.set_num_threads(old_threads)
