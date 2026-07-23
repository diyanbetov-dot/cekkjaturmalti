import gzip
import json
import shutil
import subprocess
from pathlib import Path
import tools.setup_korpus_malti as setup

archive = Path(r'C:\Users\diyan\Downloads\All.zip')
out_dir = Path('Essentials/corpus')
out_dir.mkdir(parents=True, exist_ok=True)
raw_dir = Path('.corpus_cache/local_malti_stream')
shutil.rmtree(raw_dir, ignore_errors=True)
raw_dir.mkdir(parents=True, exist_ok=True)

seven_zip = r'C:\Program Files\7-Zip\7z.exe'
subset = [
    'malti03.academic.1.txt',
    'malti03.academic.2.txt',
    'malti03.culture.1.txt',
    'malti03.culture.2.txt',
]
for name in subset:
    subprocess.run([seven_zip, 'e', str(archive), name, '-o' + str(raw_dir), '-y'], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

processed = setup.process_vertical_or_text_files(raw_dir, min_freq=2)

with gzip.open(out_dir / 'unigrams.json.gz', 'wt', encoding='utf-8') as f:
    json.dump(processed['unigrams'], f)

with gzip.open(out_dir / 'bigrams.json.gz', 'wt', encoding='utf-8') as f:
    json.dump(processed['bigrams'], f)

with gzip.open(out_dir / 'trigrams.json.gz', 'wt', encoding='utf-8') as f:
    json.dump(processed['trigrams'], f)

meta = {
    'corpus_name': 'Korpus Malti',
    'corpus_source': 'MLRS',
    'corpus_revision': 'local-archive-subset',
    'corpus_version': '4.2',
    'selected_section': 'All Sections',
    'source_url': 'file:///C:/Users/diyan/Downloads/All.zip',
    'download_timestamp': '2026-07-23T00:00:00+00:00',
    'preprocessing_version': setup.PREPROCESSING_VERSION,
    'build_date': '2026-07-23T00:00:00+00:00',
    'index_format_version': '1.0',
    'min_freq': 2,
    'stats': processed['stats'],
    'checksums': {},
    'attribution': 'Korpus Malti v4.2 provided by MLRS (Maltese Language Resource Server), University of Malta.'
}
(out_dir / 'meta.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps({'stats': processed['stats'], 'unigrams': len(processed['unigrams']), 'bigrams': len(processed['bigrams']), 'trigrams': len(processed['trigrams'])}, ensure_ascii=False))
