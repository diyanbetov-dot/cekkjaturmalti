import sys
from Essentials.app import spellchecker

word = 'iddecidew'
has_sfx = spellchecker.suffix_generator.has_suffix_parse(word)
exact_sfx = spellchecker.suffix_generator.exact_suffix_matches(word)
cs = spellchecker.suffix_generator.correct_suffix(word)

sys.stdout.buffer.write(f'has_suffix_parse: {has_sfx}\n'.encode('utf-8'))
sys.stdout.buffer.write(f'exact_suffix_matches count: {len(exact_sfx)}\n'.encode('utf-8'))
for m in exact_sfx[:5]:
    sys.stdout.buffer.write(f'  {m}\n'.encode('utf-8'))
sys.stdout.buffer.write(f'correct_suffix: {cs}\n'.encode('utf-8'))

# Also check nivvotta candidate count breakdown
for w2 in ['nivvotta', 'Amy']:
    n2 = spellchecker._normalize_word(w2)
    anchors = spellchecker._lookup_anchors(n2)
    pc = []
    for a in anchors:
        forms = spellchecker.anchor_map.get(a, [])
        for form in forms:
            keys = spellchecker.word_tags.get(form, set())
            for k in keys:
                pc.extend(spellchecker.paradigm_forms.get(k, []))
    sys.stdout.buffer.write(f'\n{w2}: anchor={anchors} paradigm_cands={len(pc)}\n'.encode('utf-8'))
