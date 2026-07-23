import importlib.util
for mod in ['zstandard', 'py7zr', 'rarfile', 'patoolib', 'libarchive']:
    print(mod, bool(importlib.util.find_spec(mod)))
