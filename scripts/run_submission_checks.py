"""CPU software checks; no pretrained model downloads or benchmark runs."""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {".git", ".venv", "outputs", "runs", "__pycache__", ".pytest_cache"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bash", help="Explicit Bash executable (e.g. Git Bash on Windows)")
    args = parser.parse_args()
    python_files = [p for p in ROOT.rglob("*.py") if not set(p.relative_to(ROOT).parts) & EXCLUDED]
    for path in python_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    print(f"Python syntax: {len(python_files)} passed", flush=True)
    shell_files = [p for p in ROOT.rglob("*.sh") if not set(p.relative_to(ROOT).parts) & EXCLUDED]
    bash = args.bash or (shutil.which("bash") if os.name != "nt" else None)
    if bash:
        for path in shell_files:
            subprocess.run([bash, "-n", str(path)], cwd=ROOT, check=True)
        print(f"Bash syntax: {len(shell_files)} passed", flush=True)
    else:
        print("Bash syntax: SKIPPED (supply --bash to check shell launchers)", flush=True)
    env = dict(os.environ, USE_TF="0", USE_FLAX="0", PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false")
    env["PYTHONPATH"] = str(ROOT / "src")
    for key in tuple(env):
        if key.startswith("TS_GROUNDER_"):
            env.pop(key)
    with tempfile.TemporaryDirectory(prefix="tango-software-check-") as directory:
        work = Path(directory)
        subprocess.run([sys.executable, "-B", "-m", "pytest", str(ROOT / "tests"), "-q",
                        "-p", "no:cacheprovider", "--basetemp", str(work / "pytest"), "--tb=short"],
                       cwd=ROOT, env=env, check=True)
        output = work / "prepare"
        subprocess.run([sys.executable, "-B", str(ROOT / "train.py"), "--config",
                        str(ROOT / "configs/vlm_smoke.yaml"), "--mode", "prepare_only",
                        "--output-dir", str(output)], cwd=ROOT, env=env, check=True)
        summary = json.loads((output / "prepare_summary.json").read_text(encoding="utf-8"))
        oracle = summary["oracle_metrics"]
        if oracle["num_grounding_examples"] <= 0 or oracle["metrics"]["parse_success_rate"] != 1.0:
            raise RuntimeError("Miniature data/parser smoke check failed")
    print("Software checks passed. Oracle checks are parser checks, not model accuracy. No pretrained/GPU benchmark was run.")


if __name__ == "__main__":
    main()
