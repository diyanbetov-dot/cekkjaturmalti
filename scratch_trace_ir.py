import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from Essentials import app
sc = app.spellchecker

# Trace the article-like token check and correct_word for 'ir' and 'ix'
for word in ["ir", "ix", "talba", "mit"]:
    norm = sc._normalize_word(word)
    is_art = sc._article_like_token(word)
    corrected = sc.correct_word(word)
    in_dict = norm in sc.dictionary_set
    tags = sc._word_tag_markers(norm) if in_dict else set()
    print(f"word='{word}' | norm='{norm}' | in_dict={in_dict} | is_article_like={is_art} | corrected='{corrected}' | tags={tags}")
