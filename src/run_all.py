"""Run the complete Dynamic LLM Execution & Triton Sparse FFN experiment suite sequentially."""

import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent

EXPERIMENTS = [
    "exp1_oracle.py",
    "exp2_temporal.py",
    "exp3_predictor.py",
    "plot_results.py",
    "exp4_markov_outliers.py",
    "exp5_compensated_adaptive.py",
    "exp6_precision_routers.py",
    "exp7_triton_benchmark.py",
]


def run_step(script_name: str):
    print(f"\n{'=' * 72}\n>>> RUNNING {script_name}\n{'=' * 72}", flush=True)
    subprocess.run([sys.executable, str(SRC_DIR / script_name)], check=True)


def main():
    for script in EXPERIMENTS:
        run_step(script)
    print("\nALL EXPERIMENTS (1..7) COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
