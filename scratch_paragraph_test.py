import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from Essentials import app

sc = app.spellchecker

text = (
    "bongu lil kulhadd.  sfortunatament kelli nikteb hawn biex tkunu tafu kif tnejjek bina il gvern fuq ir 'remote working'.  "
    "il mara tieghi kisret saqajha u talbet lil 'ceo' tal 'poyc' biex tahdem mid dar.  "
    "wara gimgha mit talba qalulha li trid tohodhom sick.  "
    "issa f dan id departiment lil min irridu jghatu 'remote working' lil tal qalba imma lilna le.  "
    "Nistaqsi lil prim ministru u lid deputat prim ministru fej hu dan id dritt.  "
    "il mara saqsiet ghall remote working sakemm jghaddilha u mhux b mod permanenti.  "
    "haga ohra hi din li morna ngibu krozzi minn mater dei u qas ghandhom.  "
    "igiefieri il mara qas kieku tipprova tmur ix xoghol ma tista."
)

print("=== INPUT ===")
print(text)
print()

try:
    result = sc.correct_text_rich(text, edit_distance_tolerance=1)
    print("=== OUTPUT ===")
    print(result["corrected_text"])
    print()
    print("=== TOKEN ISSUES (ambiguous/crucial only) ===")
    for tok in result["tokens"]:
        if tok.get("ambiguous") or tok.get("crucial"):
            orig = tok.get("original", "")
            corr = tok.get("corrected", "")
            choices = [c.get("word","") for c in tok.get("choices", [])]
            print(f"  [{tok['type']}] '{orig}' -> '{corr}' | choices: {choices}")
except Exception as e:
    import traceback
    print("ERROR:", e)
    traceback.print_exc()
