import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

app_path = r'C:\Users\diyan\OneDrive\New folder\Desktop\Ċekkjatur tal-Kitba\Essentials\app.py'
with open(app_path, 'r', encoding='utf-8') as f:
    content = f.read()

replacements = [
    (
        '''        "terbahx": "tirbaħx",
        "f'ghoxx": "f'għoxx",
    }''',
        '''        "terbahx": "tirbaħx",
        "f'ghoxx": "f'għoxx",
        "malta": "Malta",
        "qazzist": "qażżiżt",
        "qazzisti": "qażżiżti",
    }'''
    ),
    (
        '''    def _simple_noun_possessive_surface_base(self, word: str) -> str | None:
        normalized = self._normalize_word(word)
        suffixes = ("hom", "kom", "na", "ha", "ek", "ok", "u", "i", "k", "h")
        for suffix in suffixes:
            if not normalized.endswith(suffix) or len(normalized) <= len(suffix) + 1:
                continue
            base = normalized[: -len(suffix)]
            if base in self.dictionary_set and self._is_probable_noun(base):
                return base
        return None''',
        '''    def _simple_noun_possessive_surface_base(self, word: str) -> str | None:
        normalized = self._normalize_word(word)
        suffixes = ("hom", "kom", "na", "ha", "ek", "ok", "u", "i", "k", "h")
        for suffix in suffixes:
            if not normalized.endswith(suffix) or len(normalized) <= len(suffix) + 1:
                continue
            base = normalized[: -len(suffix)]
            if base in self.dictionary_set and self._is_probable_noun(base):
                return base
            
            if len(base) >= 2:
                base_e = base[:-1] + "e" + base[-1]
                if base_e in self.dictionary_set and self._is_probable_noun(base_e):
                    return base_e
                base_i = base[:-1] + "i" + base[-1]
                if base_i in self.dictionary_set and self._is_probable_noun(base_i):
                    return base_i
        return None'''
    ),
    (
        '''        validated_existing_initial_surface = self._validated_existing_initial_vowel_surface(
            normalized
        )
        if validated_existing_initial_surface is not None:
            return self._store_correct_word_result(
                word,
                self._match_capitalisation(word, validated_existing_initial_surface),
                is_deterministic=True,
            )''',
        '''        validated_existing_initial_surface = self._validated_existing_initial_vowel_surface(
            normalized
        )
        if validated_existing_initial_surface is not None:
            if normalized.startswith(("i", "u")) and validated_existing_initial_surface.startswith(("j", "w")):
                prefer_plain, prefer_vowel = self._initial_vowel_surface_options(validated_existing_initial_surface)
                if prefer_vowel:
                    validated_existing_initial_surface = prefer_vowel[0]
            elif normalized.startswith(("j", "w")) and validated_existing_initial_surface.startswith(("i", "u")):
                prefer_plain, prefer_vowel = self._initial_vowel_surface_options(validated_existing_initial_surface)
                if prefer_plain:
                    validated_existing_initial_surface = prefer_plain[0]
            
            return self._store_correct_word_result(
                word,
                self._match_capitalisation(word, validated_existing_initial_surface),
                is_deterministic=True,
            )'''
    )
]

for old, new in replacements:
    if old not in content:
        print(f"FAILED to find:\n{old}")
        sys.exit(1)
    content = content.replace(old, new)

with open(app_path, 'w', encoding='utf-8') as f:
    f.write(content)
print("SUCCESS")
