import cProfile
import pstats
import io
from Essentials.app import spellchecker

pr = cProfile.Profile()
pr.enable()

for _ in range(50):
    spellchecker.suggest('bahar')

pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
ps.print_stats(30)
print(s.getvalue())
