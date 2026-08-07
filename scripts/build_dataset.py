import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from neural.dataset import prepare_processed_datasets

if __name__ == "__main__":
    n_train, n_val, n_test = prepare_processed_datasets()
    print(f"Dataset prepared: {n_train} train, {n_val} val, {n_test} test.")
