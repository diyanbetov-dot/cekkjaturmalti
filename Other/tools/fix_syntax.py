import re

with open('app.py', 'r', encoding='utf-8') as f:
    text = f.read()

# We need to fix the syntax error at line 418
old_text = """        return normalized.startswith(
            (
                "a",
                "e",
                "i",
                "o",
                "u",
        "lu",
        "la",
        "lek",
        "lok",
        "lkom",
        "lna",
        "lhom",
        "uli",
        "ulu",
        "ula",
    )

    MANUAL_ENDING_REPAIRS = ("""

new_text = """        return normalized.startswith(
            (
                "a",
                "e",
                "i",
                "o",
                "u",
                "à",
                "è",
                "ì",
                "ò",
                "ù",
                "għ",
                "gh",
                "h",
                "ħ",
            )
        )

    MANUAL_EJD_AJD_TAILS = (
        "",
        "u",
        "x",
        "la",
        "lek",
        "lok",
        "lkom",
        "lna",
        "lhom",
        "uli",
        "ulu",
        "ula",
    )

    MANUAL_ENDING_REPAIRS = ("""

fixed_text = text.replace(old_text, new_text)

if fixed_text != text:
    print('Replaced exact')
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(fixed_text)
else:
    print('Failed to replace exact')
