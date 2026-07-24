from Essentials import app

text = """bongu lil kulhadd.  sfortunatament kelli nikteb hawn biex tkunu tafu kif tnejjek bina il gvern fuq ir 'remote working'.  il mara tieghi kisret saqajha u talbet lil 'ceo' tal 'poyc' biex tahdem mid dar.  wara gimgha mit talba qalulha li trid tohodhom sick.  issa f dan id departiment lil min irridu jghatu 'remote working' lil tal qalba imma lilna le.  Nistaqsi lil prim ministru u lid deputat prim ministru fej hu dan id dritt.  il mara saqsiet ghall remote working sakemm jghaddilha u mhux b mod permanenti.  haga ohra hi din li morna ngibu krozzi minn mater dei u qas ghandhom.  igiefieri il mara qas kieku tipprova tmur ix xoghol ma tista."""

print("STAGE 1: quoted English")
print(
    "remote working accepted:",
    app.spellchecker._accepted_exact_english("remote working"),
)

print("\nSTAGE 2: correct_text_rich")
result = app.spellchecker.correct_text_rich(
    text,
    edit_distance_tolerance=1,
)
print(result["corrected_text"])

print("\nSTAGE 3: grammar analysis")
corrected_text = result["corrected_text"]
tokens = result["tokens"]

request_words = [
    match.group(0)
    for match in app.spellchecker.WORD_PATTERN.finditer(corrected_text)
]

findings = app.grammar_rule_engine.analyze(
    text=corrected_text,
    request_words=request_words,
    tokens=tokens,
)

corrected_text, tokens, _ = app.grammar_rule_engine.apply_safe_rewrites(
    original_text=corrected_text,
    corrected_text=corrected_text,
    tokens=tokens,
)

print(corrected_text)
print("\nAll stages completed.")
