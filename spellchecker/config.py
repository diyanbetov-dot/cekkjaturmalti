import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

BERTU_MODEL_ID = os.environ.get("BERTU_MODEL_ID", "MLRS/BERTu")
BERTU_MAX_CONTEXT_TOKENS = 384
MAX_TOKEN_CANDIDATES = 24
MAX_SPAN_CANDIDATES = 12
MAX_BERTU_CANDIDATES = 4

FINAL_DICS_DIR = BASE_DIR / "Essentials" / "finaldics"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
DATA_ARTIFACTS_DIR = BASE_DIR / "data" / "artifacts"
MODELS_HYBRID_DIR = BASE_DIR / "models" / "hybrid"

AI_CORRECTIONS_FILE = BASE_DIR / "AI corrections.txt"

# Ensure directories exist
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
DATA_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_HYBRID_DIR.mkdir(parents=True, exist_ok=True)
