with open('Essentials/app.py', 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f, 1):
        if 'Re-resolve' in line:
            print(f"Line {idx} matches:")
            print(f"  {repr(line)}")
