"""
VITALSIGN HEALTHCARE PREDICTION
Regularizatio_ridge_laso_elastic_M2.py
Wraps Regularization_ridge_laso_elastic_M2.py.
"""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).parent / "Regularization_ridge_laso_elastic_M2.py"
    subprocess.run([sys.executable, str(target)], check=True)
