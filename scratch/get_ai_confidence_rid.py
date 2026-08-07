"""scratch/get_ai_confidence_rid.py
Extract exact AI character action probabilities for 'rid' -> 'rrid'.
INSPECT ONLY.
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from neural_corrector.inference.corrector import NeuralCorrector
from neural_corrector.models.alignment import COPY_ACTION

text = "manafx xha naqbad namel andi hafna dwejjaq li rid nehles minnu."
DEFAULT_ARTIFACT = Path("neural_corrector/artifacts/char_edit_bigru_v4")

nc = NeuralCorrector(DEFAULT_ARTIFACT)

print("=" * 80)
print(f"INPUT SENTENCE:\n  {text}")
print("=" * 80)

# Run model raw predictions to get probabilities
raw_text = text
chars = list(raw_text)
indices = [nc.vocab.characters.get(c, nc.vocab.characters.get("<UNK>", 1)) for c in chars]
tensor_in = torch.tensor([indices], dtype=torch.long, device=nc.device)

lengths = torch.tensor([len(chars)], dtype=torch.long, device=nc.device)
with torch.no_grad():
    logits = nc.model(tensor_in, lengths)[0]  # [seq_len, vocab_size]
    probs = torch.softmax(logits, dim=-1)

rid_start = raw_text.find("rid")
print(f"\nTarget word 'rid' found at character index {rid_start} in full sentence:")

for i in range(rid_start, rid_start + len("rid")):
    char = raw_text[i]
    top_probs, top_indices = torch.topk(probs[i], k=5)
    print(f"\nPosition {i} (char: '{char}'):")
    for prob, idx in zip(top_probs, top_indices):
        action_name = nc.vocab.inverse_actions[idx.item()] if idx.item() < len(nc.vocab.inverse_actions) else f"idx_{idx.item()}"
        print(f"  Action '{action_name:<12}': probability = {prob.item():.6f}")

# Action required to insert 'r' before 'r':
# Usually "rKEEP" or "KEEP"
print("\n" + "=" * 80)
print("ANALYSIS SUMMARY:")
print("Action threshold required for neural replacement:", nc.threshold)
