MALTESE_VOWELS = set("aeiouàèìòùáéíóúâêîôû")
SUN_CONSONANTS = {"d", "n", "r", "s", "t", "x", "z", "ċ", "ż"}
BASE_PREPS = {
    "għa": "għall",
    "mi": "mill",
    "sa": "sal",
    "ma": "mal",
    "ta": "tal",
    "bi": "bil",
    "fi": "fil",
    "bħa": "bħall"
}

ALL_FORMS = {"il", "l", "'il", "'l", "\u2019il", "\u2019l"}
ALL_FORMS.update(BASE_PREPS.values())
for stem in BASE_PREPS.keys():
    ALL_FORMS.update(stem + c for c in SUN_CONSONANTS)
ALL_FORMS.update("i" + c for c in SUN_CONSONANTS)

def test(word, phrase):
    current_norm = word.lower()
    if current_norm not in ALL_FORMS:
        return word
    first_char = phrase[0].lower() if phrase else ""
    
    prefix_char = ""
    if current_norm.startswith("'") or current_norm.startswith("\u2019"):
        prefix_char = current_norm[0]
        current_norm = current_norm[1:]
    
    word_stem = ""
    for s in BASE_PREPS:
        if current_norm.startswith(s):
            word_stem = s
            break
            
    if first_char in MALTESE_VOWELS:
        expected = BASE_PREPS[word_stem] if word_stem else "l"
    elif first_char in SUN_CONSONANTS:
        expected = word_stem + first_char if word_stem else "i" + first_char
    else:
        expected = BASE_PREPS[word_stem] if word_stem else "il"
        
    return prefix_char + expected + "-"

print("tal + washing ->", test("tal", "washing machine"))
print("il + washing ->", test("il", "washing machine"))
print("il + apple ->", test("il", "apple"))
print("tal + supermarket ->", test("tal", "supermarket"))
print("fil + server ->", test("fil", "server"))
print("is + server ->", test("is", "server"))
print("mal + drone ->", test("mal", "drone"))
print("għall + table ->", test("għall", "table"))
print("tas + shop ->", test("tas", "shop"))
print("'il + apple ->", test("'il", "apple"))
print("'l + apple ->", test("'l", "apple"))
