# -*- coding: utf-8 -*-
import json, sys
sys.stdout.reconfigure(encoding="utf-8")

with open("scratch_main_out.json", "r", encoding="utf-8") as f:
    main_data = json.load(f)

with open("scratch_experimental_enabled_out.json", "r", encoding="utf-8") as f:
    exp_data = json.load(f)

main_tokens = main_data["tokens"]
exp_tokens = exp_data["tokens"]

print(f"Total tokens in Main: {len(main_tokens)}")
print(f"Total tokens in Experimental: {len(exp_tokens)}")

print("\n--- ANNOTATED TOKEN-BY-TOKEN COMPARISON ---")

diff_count = 0
for idx, (mt, et) in enumerate(zip(main_tokens, exp_tokens)):
    orig = mt.get("original", "")
    mc = mt.get("corrected", "")
    ec = et.get("corrected", "")
    
    if mc != ec:
        diff_count += 1
        print(f"[{diff_count}] Token '{orig}':")
        print(f"    - Main Branch:         '{mc}'")
        print(f"    - Experimental Branch: '{ec}'")

if diff_count == 0:
    print("Both branches produced identical token-level corrections for this text input.")
