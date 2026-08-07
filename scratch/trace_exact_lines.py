# -*- coding: utf-8 -*-
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from Essentials.app import spellchecker
import trace

tracer = trace.Trace(count=False, trace=True)

def run():
    spellchecker.correct_text_rich("sal bahar")

# Filter trace to only spellchecker.py
tracer.runfunc(run)
