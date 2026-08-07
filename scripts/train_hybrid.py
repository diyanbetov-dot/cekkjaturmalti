import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from neural.train import train_and_save_hybrid_model

if __name__ == "__main__":
    heads_path, manifest_path = train_and_save_hybrid_model()
    print(f"Model heads saved to {heads_path}")
    print(f"Manifest saved to {manifest_path}")
