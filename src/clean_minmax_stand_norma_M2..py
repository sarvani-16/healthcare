"""
VITALSIGN HEALTHCARE PREDICTION
clean_minmax_stand_norma_M2..py
Wraps clean_minmax_stand_norma_M2.py.
"""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).parent / "clean_minmax_stand_norma_M2.py"
    subprocess.run([sys.executable, str(target)], check=True)
