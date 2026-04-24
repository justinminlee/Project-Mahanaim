"""
One-click setup script.
Run: python setup.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def run(cmd: list[str], desc: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"❌ Failed: {' '.join(cmd)}")
        sys.exit(1)
    print(f"✅ Done: {desc}")


def main() -> None:
    print("\n🔐 FraudGuard – Setup Script")
    print("Setting up Credit Card Fraud Detection Dashboard\n")

    # 1. Install Python dependencies
    run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"],
        "Installing Python dependencies",
    )

    # 2. Generate synthetic dataset
    run(
        [sys.executable, "data/generate_data.py"],
        "Generating synthetic transaction dataset (50,000 rows)",
    )

    # 3. Train model
    run(
        [sys.executable, "models/train_model.py"],
        "Training Random Forest fraud detection model",
    )

    print("\n" + "="*60)
    print("🎉 Setup complete!")
    print("="*60)
    print("\nTo launch the dashboard, run:")
    print("  streamlit run dashboard/app.py\n")


if __name__ == "__main__":
    main()
