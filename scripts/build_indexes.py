import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from lexicon.indexes import LexiconIndexes
from spellchecker.config import FINAL_DICS_DIR

if __name__ == "__main__":
    indexes = LexiconIndexes(FINAL_DICS_DIR)
    print(f"Loaded {len(indexes.word_map)} dictionary surface records.")
