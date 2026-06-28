import tracemalloc
import sys
tracemalloc.start()
from Essentials.app import spellchecker
snapshot = tracemalloc.take_snapshot()
stats = snapshot.statistics('lineno')
total = sum(s.size for s in stats)
sys.stdout.buffer.write(f'Total memory after init: {total / 1024 / 1024:.1f} MB\n'.encode('utf-8'))
top = sorted(stats, key=lambda x: x.size, reverse=True)[:15]
for s in top:
    line = f'{s.size / 1024 / 1024:.2f} MB | {s.count} obj | {s.traceback.format()[0]}\n'
    sys.stdout.buffer.write(line.encode('utf-8'))
