# -*- coding: utf-8 -*-
import json, sys
sys.stdout.reconfigure(encoding="utf-8")

with open("scratch_main_out.json", "r", encoding="utf-8") as f:
    main_data = json.load(f)

with open("scratch_experimental_out.json", "r", encoding="utf-8") as f:
    exp_data = json.load(f)

main_text = main_data["corrected_text"]
exp_text = exp_data["corrected_text"]

print("--- MAIN BRANCH OUTPUT ---")
print(main_text)
print("\n--- EXPERIMENTAL BRANCH OUTPUT ---")
print(exp_text)

print("\n--- DIFFERENCES ---")
main_lines = main_text.splitlines()
exp_lines = exp_text.splitlines()

for idx, (ml, el) in enumerate(zip(main_lines, exp_lines)):
    if ml != el:
        print(f"Line {idx+1}:")
        print(f"  MAIN: {ml}")
        print(f"  EXP:  {el}")
