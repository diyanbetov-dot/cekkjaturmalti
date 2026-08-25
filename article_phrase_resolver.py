from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path

from corpus_ranker import CorpusCandidateRanker, CorpusEvidence


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value).casefold().strip()
    return normalized.replace("’", "'")


@dataclass(frozen=True, slots=True)
class ArticleForm:
    surface: str
    pos: str
    article_letter: str


@dataclass(frozen=True, slots=True)
class ArticleResolution:
    joined_prefix: str
    separate_prefix: str
    article_tag: str
    ambiguous: bool
    corpus_evidence: CorpusEvidence
    separate_bigram: float


class ArticlePhraseResolver:
    """Resolve unfinished article/preposition forms over a two-word span."""

    COMPLEMENT_TAGS = (
        "NOUN",
        "ADJ",
        "PLACE",
        "NAME",
        "SNAME",
        "PRON",
        "CARDNUM",
        "ORDNUM",
    )
    NO_NOUN_FOLLOWING = {"id", "għar"}
    # A canonical bare ġo is explicit. Its article paradigm is considered
    # only when the input itself contains article material.
    EXPLICIT_BARE_ONLY = {"ġo"}

    def __init__(
        self,
        dictionary_dir: Path,
        corpus_ranker: CorpusCandidateRanker,
        *,
        enabled: bool = True,
    ) -> None:
        self.dictionary_dir = Path(dictionary_dir)
        self.corpus_ranker = corpus_ranker
        self.enabled = bool(enabled)
        self.forms: dict[str, tuple[ArticleForm, ...]] = {}
        self.tags: dict[str, set[str]] = {}
        self.nonarticle_surfaces: set[str] = set()
        self.bare_forms: set[str] = set()
        self.short_l_forms: tuple[str, ...] = ()
        self.canonical_form_count = 0
        if self.enabled:
            self._load()

    @staticmethod
    def _entry_parts(line: str) -> tuple[str, str] | None:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "/" not in stripped:
            return None
        surface, payload = stripped.split("/", 1)
        return surface.strip(), payload.strip()

    def _load(self) -> None:
        articles_path = self.dictionary_dir / "articles.dic"
        if not articles_path.exists():
            return

        article_rows: dict[str, list[ArticleForm]] = {}
        current_bare: tuple[str, str] | None = None
        simple_articles: list[ArticleForm] = []
        for line in articles_path.read_text(encoding="utf-8-sig").splitlines():
            parts = self._entry_parts(line)
            if parts is None:
                continue
            surface, payload = parts
            fields = payload.split("-")
            if len(fields) == 2 and fields[1] == "NO":
                current_bare = (_normalize(surface), fields[0])
                self.bare_forms.add(current_bare[0])
                continue
            if len(fields) != 3 or fields[1] != "YES" or not surface.endswith("-"):
                continue
            form = ArticleForm(surface=surface, pos=fields[0], article_letter=fields[2])
            self.canonical_form_count += 1
            article_rows.setdefault(_normalize(surface[:-1]), []).append(form)
            if current_bare is not None and current_bare[1] == form.pos:
                article_rows.setdefault(current_bare[0], []).append(form)
            elif form.pos == "DET" and current_bare is not None and current_bare[1] == "PREP":
                simple_articles.append(form)
        for alias in ("l", "il", "'l", "'l-", "’l", "’l-", "s"):
            article_rows.setdefault(alias, []).extend(simple_articles)
        # Detached assimilated articles are common informal input: d dinja,
        # t tifel, r raġel. They map back to their full id-/it-/ir- entries;
        # sentence phonology later decides whether the initial i is elided.
        for form in simple_articles:
            normalized_surface = _normalize(form.surface)
            if (
                form.pos == "DET"
                and normalized_surface.startswith("i")
                and normalized_surface.endswith("-")
                and len(normalized_surface) == 3
            ):
                article_rows.setdefault(normalized_surface[1], []).append(form)
        # "mal" is a common written variant of "ma'" used before the article.
        # Copy all ma' forms so that "mal hanut" resolves to "mal-ħanut" etc.
        if "ma'" in article_rows:
            article_rows.setdefault("mal", list(article_rows["ma'"]))
        elif "ma" in article_rows:
            article_rows.setdefault("mal", list(article_rows["ma"]))
        self.forms = {key: tuple(value) for key, value in article_rows.items()}

        short_l_forms: list[str] = []
        prepositions_path = self.dictionary_dir / "prepositions.dic"
        if prepositions_path.exists():
            for line in prepositions_path.read_text(encoding="utf-8-sig").splitlines():
                parts = self._entry_parts(line)
                if parts is None:
                    continue
                surface, payload = parts
                tag = payload.split("-", 1)[0]
                normalized_surface = _normalize(surface)
                if tag not in {"SHORTDEFPREP", "ISHORTDEFPREP"}:
                    continue
                if normalized_surface.rstrip("-") not in {"'l", "'il"}:
                    continue
                if normalized_surface not in short_l_forms:
                    short_l_forms.append(normalized_surface)
        self.short_l_forms = tuple(short_l_forms)

        for path in sorted(self.dictionary_dir.glob("*.dic")):
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                parts = self._entry_parts(line)
                if parts is None:
                    continue
                surface, payload = parts
                key = _normalize(surface)
                self.tags.setdefault(key, set()).add(payload.split("-", 1)[0])
                if path != articles_path:
                    self.nonarticle_surfaces.add(key)

    @staticmethod
    def _initial_letter(word: str) -> str:
        normalized = _normalize(word).lstrip("'’")
        return normalized[:1]

    @staticmethod
    def _uses_neutral_l_article(initial: str) -> bool:
        if not initial:
            return False
        assimilating = {"ċ", "d", "n", "r", "s", "t", "x", "z", "ż"}
        return initial not in assimilating

    def _is_complement(self, word: str) -> bool:
        tags = self.tags.get(_normalize(word), set())
        if not tags:
            # Unknown words still receive safe surface article normalization.
            return True
        return any(any(marker in tag for marker in self.COMPLEMENT_TAGS) for tag in tags)

    def _is_noun(self, word: str) -> bool:
        return any("NOUN" in tag for tag in self.tags.get(_normalize(word), set()))

    def resolve(
        self,
        prefix: str,
        following: str,
        *,
        previous: str | None = None,
        after: str | None = None,
    ) -> ArticleResolution | None:
        if not self.enabled or not following:
            return None

        explicit_hyphen = prefix.endswith("-")
        key = _normalize(prefix).removesuffix("-")

        if key in self.EXPLICIT_BARE_ONLY and not explicit_hyphen:
            return None

        # Demonstrative double-article guard: e.g. "Din in-Nanna" -> don't join "Din" with "in-Nanna"
        article_tokens = {"il", "in", "iċ", "id", "ir", "is", "it", "ix", "iz", "iż", "l"}
        following_norm = _normalize(following)
        if key in ("din", "dan", "dawk") and (
            following_norm in article_tokens
            or "-" in following
            or following_norm.startswith(tuple(article + "-" for article in article_tokens))
        ):
            return None

        required_initial = self._initial_letter(following)
        following_tags = self.tags.get(_normalize(following), set())
        proper_following = bool(
            following[:1].isupper()
            and (
                not following_tags
                or any(any(marker in tag for marker in ("NAME", "SNAME", "PLACE")) for tag in following_tags)
            )
        )
        if key in self.bare_forms and proper_following and not explicit_hyphen:
            return None
        if key in {"dan", "din", "dawk"} and any(
            tag in {"PRON", "T", "AS", "Q"} for tag in following_tags
        ):
            return None
        vowels = ("a", "e", "i", "o", "u", "à", "è", "ì", "ò", "ù")
        matches = [
            form for form in self.forms.get(key, ())
            if form.article_letter == required_initial
            or (form.article_letter == "l" and self._uses_neutral_l_article(required_initial))
        ]
        if not matches or not self._is_complement(following):
            return None

        form = matches[0]
        joined_prefix = form.surface
        previous_norm = _normalize(previous or "")
        previous_ends_vowel = bool(previous_norm and previous_norm[-1:] in vowels)
        if (
            len(key) == 1
            and joined_prefix == f"i{key}-"
            and previous_ends_vowel
        ):
            joined_prefix = f"{key}-"
        separate_prefix = prefix.removesuffix("-")
        if key == "min":
            separate_prefix = "minn"
        elif key in ("ghal", "għal"):
            separate_prefix = "għal"

        collision = key in self.bare_forms
        # lil + an ordinary noun takes the definite article (lill-). Proper
        # names and generated possessive forms are handled separately by the
        # caller and retain bare lil.
        force_join_noun = key == "lil" and self._is_noun(following)
        if force_join_noun:
            collision = False
        if explicit_hyphen or (key in self.NO_NOUN_FOLLOWING and self._is_noun(following)):
            collision = False

        # The current corpus indexes words without article hyphens. They can
        # still provide surrounding evidence and flag an attested separate pair,
        # but structural compatibility remains the primary signal.
        evidence = self.corpus_ranker.evidence(
            key,
            previous=previous,
            following=following,
        )
        separate_bigram = 0.0
        if self.corpus_ranker.available:
            separate_bigram = float(
                self.corpus_ranker.bigrams.get(key, {}).get(_normalize(following), 0.0)
            )

        is_vowel = required_initial in vowels
        # A separated definite article is not a genuine lexical alternative:
        # ``it tifel``, ``id dar`` and ``il verita`` must surface with a
        # hyphen. Corpus tokenisation frequently contains the same missing-
        # hyphen typo, so a corpus bigram must never turn DET joining into an
        # ambiguity. PREP/DET collisions remain context-resolvable.
        ambiguous = False
        if form.pos != "DET" and key in self.bare_forms:
            ambiguous = (
                not force_join_noun
                and (
                    (collision and not is_vowel)
                    or (separate_bigram > 0.0 and not explicit_hyphen)
                    or (key in ("min", "ghal", "għal") and not explicit_hyphen)
                )
            )

        return ArticleResolution(
            joined_prefix=joined_prefix,
            separate_prefix=separate_prefix,
            article_tag=f"{form.pos}-YES-{form.article_letter}",
            ambiguous=ambiguous,
            corpus_evidence=evidence,
            separate_bigram=separate_bigram,
        )


    def status_payload(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "forms": self.canonical_form_count,
            "surface_keys": len(self.forms),
        }
