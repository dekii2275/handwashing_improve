"""
Train and Evaluate Reproduced Handwashing Models on RTX 3050 GPU.
"""

import sys
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_DIR = PROJECT_ROOT / "scratch" / "friend_repo"
CONFIG_PATH = PROJECT_ROOT / "configs" / "train_reproduced.yaml"
DATA_DIR = PROJECT_ROOT / "data" / "processed_w48_s4"
OUTPUT_DIR = PROJECT_ROOT / "models" / "gru_shape_relation_motion"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cmd = [
    sys.executable, "-u", str(REPO_DIR / "train.py"),
    "--config", str(CONFIG_PATH),
    "--data-dir", str(DATA_DIR),
    "--output-dir", str(OUTPUT_DIR),
    "--groups", "shape", "relation", "motion",
    "--evaluate-test",
    "--resume", "auto",
]

print("Executing command:")
print(" ".join(cmd))

process = subprocess.Popen(
    cmd,
    cwd=str(REPO_DIR),
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)

for line in process.stdout:
    print(line, end="", flush=True)

ret = process.wait()
if ret != 0:
    raise RuntimeError(f"Training failed with code {ret}")
print(f"\nTraining completed successfully! Artifacts in: {OUTPUT_DIR}")
