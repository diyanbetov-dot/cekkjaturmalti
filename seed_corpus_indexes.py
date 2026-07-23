import json, gzip
from pathlib import Path
out = Path('Essentials/corpus')
out.mkdir(parents=True, exist_ok=True)
with gzip.open(out / 'unigrams.json.gz', 'wt', encoding='utf-8') as f:
    json.dump({'jien': 3.0, 'tgħid': 2.0, 'mela': 1.0}, f)
with gzip.open(out / 'bigrams.json.gz', 'wt', encoding='utf-8') as f:
    json.dump({'jien': {'tgħid': 2.0}}, f)
with gzip.open(out / 'trigrams.json.gz', 'wt', encoding='utf-8') as f:
    json.dump({'jien jgħid': {'mela': 1.0}}, f)
(out / 'meta.json').write_text(json.dumps({
    'corpus_name': 'Korpus Malti',
    'corpus_source': 'local',
    'preprocessing_version': '1.1.0',
    'stats': {'total_tokens': 6, 'vocab_size': 3, 'bigram_count': 1, 'trigram_count': 1, 'valid_rows': 6, 'malformed_rows': 0},
    'build_date': '2026-07-22T00:00:00+00:00'
}, indent=2), encoding='utf-8')
print('wrote', out)
