"""
VITALSIGN HEALTHCARE PREDICTION
Module M1: load_understand_dataset.py
Wraps load_understand_dataset_M1.py for seamless execution.
"""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).parent / "load_understand_dataset_M1.py"
    subprocess.run([sys.executable, str(target)], check=True)
