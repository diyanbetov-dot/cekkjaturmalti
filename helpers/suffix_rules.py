"""
suffix_rules.py
===============

Suffix ending data + forward stem transformations.

This module is deliberately not a brute-force generator.  The parser in
suffix_generator.py first decides which ending family is relevant, then this
file only generates candidates for that family.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

try:
    from .verb_form_index import VerbFormRecord, MalteseVerbFormIndex
except ImportError:  # pragma: no cover
    from verb_form_index import VerbFormRecord, MalteseVerbFormIndex


@dataclass(frozen=True)
class SuffixSpec:
    label: str
    kind: str
    person: str
    display: str
    canonical_surfaces: tuple[str, ...]
    typed_endings: tuple[str, ...]
    priority: int


@dataclass(frozen=True)
class ParsedSuffix:
    spec: SuffixSpec
    typed_ending: str
    typed_stem: str
    priority: int


@dataclass(frozen=True)
class GeneratedSuffixCandidate:
    surface: str
    base: str
    suffix_label: str
    suffix_kind: str
    suffix_person: str
    suffix_display: str
    rule_id: str
    rule_description: str
    raw_tag: str
    root: str
    form_class: str
    tense: str
    person: str
    root_class: str

    def to_debug_dict(self) -> dict:
        return {
            "surface": self.surface,
            "base": self.base,
            "suffix_label": self.suffix_label,
            "suffix_kind": self.suffix_kind,
            "suffix_person": self.suffix_person,
            "suffix_display": self.suffix_display,
            "rule_id": self.rule_id,
            "rule_description": self.rule_description,
            "raw_tag": self.raw_tag,
            "root": self.root,
            "form_class": self.form_class,
            "tense": self.tense,
            "person": self.person,
            "root_class": self.root_class,
        }


DO_SUFFIXES: tuple[SuffixSpec, ...] = (
    SuffixSpec(
        "DO_1S",
        "DO",
        "1S",
        "-ni",
        ("ni",),
        ("ni",),
        10,
    ),
    SuffixSpec(
        "DO_2S",
        "DO",
        "2S",
        "-k/-ek/-ok",
        ("k", "ek", "ok"),
        ("ek", "ok", "k"),
        20,
    ),
    SuffixSpec(
        "DO_3SM",
        "DO",
        "3SM",
        "-u/-h",
        ("u", "h"),
        ("u", "h"),
        30,
    ),
    SuffixSpec(
        "DO_3SF",
        "DO",
        "3SF",
        "-ha",
        ("ha",),
        ("ha", "wa", "ja", "a"),
        40,
    ),
    SuffixSpec(
        "DO_1P",
        "DO",
        "1P",
        "-na",
        ("na",),
        ("na",),
        50,
    ),
    SuffixSpec(
        "DO_2P",
        "DO",
        "2P",
        "-kom",
        ("kom",),
        ("kom",),
        60,
    ),
    SuffixSpec(
        "DO_3P",
        "DO",
        "3P",
        "-hom",
        ("hom",),
        ("hom", "om", "wom", "jom"),
        70,
    ),
)

IDO_SUFFIXES: tuple[SuffixSpec, ...] = (
    SuffixSpec(
        "IDO_1S",
        "IDO",
        "1S",
        "-li",
        ("li",),
        ("ili", "li"),
        110,
    ),
    SuffixSpec(
        "IDO_2S",
        "IDO",
        "2S",
        "-lek/-lok",
        ("lek", "lok"),
        ("ilek", "ilok", "lek", "lok"),
        120,
    ),
    SuffixSpec(
        "IDO_3SM",
        "IDO",
        "3SM",
        "-lu",
        ("lu",),
        ("ilu", "lu"),
        130,
    ),
    SuffixSpec(
        "IDO_3SF",
        "IDO",
        "3SF",
        "-lha",
        ("lha",),
        ("ilha", "ila", "lha"),
        140,
    ),
    SuffixSpec(
        "IDO_1P",
        "IDO",
        "1P",
        "-lna",
        ("lna",),
        ("ilna", "lna"),
        150,
    ),
    SuffixSpec(
        "IDO_2P",
        "IDO",
        "2P",
        "-lkom",
        ("lkom",),
        ("ilkom", "lkom"),
        160,
    ),
    SuffixSpec(
        "IDO_3P",
        "IDO",
        "3P",
        "-lhom",
        ("lhom",),
        ("ilhom", "lhom", "ilom", "lom", "ilħom", "lħom"),
        170,
    ),
)

# Early support for the DO + IDO note in the rule document:
#     u -> hu, h -> hu, ha -> hie
#
# These are kept as ending families, not as fully expanded brute-force forms.
COMBINED_SUFFIXES: tuple[SuffixSpec, ...] = (
    SuffixSpec(
        "COMBINED_3SM_1S",
        "DO_IDO",
        "3SM+1S",
        "-huli",
        ("huli",),
        ("huli", "uli"),
        210,
    ),
    SuffixSpec(
        "COMBINED_3SM_2S",
        "DO_IDO",
        "3SM+2S",
        "-hulek",
        ("hulek",),
        ("hulek", "ulek"),
        220,
    ),
    SuffixSpec(
        "COMBINED_3SM_3SM",
        "DO_IDO",
        "3SM+3SM",
        "-hulu",
        ("hulu",),
        ("hulu", "ulu", "hulhu", "ulhu"),
        230,
    ),
    SuffixSpec(
        "COMBINED_3SM_3SF",
        "DO_IDO",
        "3SM+3SF",
        "-hulha",
        ("hulha",),
        ("hulha", "ulha"),
        240,
    ),
    SuffixSpec(
        "COMBINED_3SM_1P",
        "DO_IDO",
        "3SM+1P",
        "-hulna",
        ("hulna",),
        ("hulna", "ulna"),
        250,
    ),
    SuffixSpec(
        "COMBINED_3SM_2P",
        "DO_IDO",
        "3SM+2P",
        "-hulkom",
        ("hulkom",),
        ("hulkom", "ulkom"),
        260,
    ),
    SuffixSpec(
        "COMBINED_3SM_3P",
        "DO_IDO",
        "3SM+3P",
        "-hulhom",
        ("hulhom",),
        ("hulhom", "ulhom", "hulom", "ulom"),
        270,
    ),
    SuffixSpec(
        "COMBINED_3P_3P",
        "DO_IDO",
        "3P+3P",
        "-homlhom",
        ("homlhom",),
        ("homlhom", "omlhom", "womlhom", "homlom", "omlom", "womlom"),
        280,
    ),
    SuffixSpec(
        "COMBINED_3P_1S",
        "DO_IDO",
        "3P+1S",
        "-homli",
        ("homli",),
        ("homli", "omli", "womli"),
        281,
    ),
    SuffixSpec(
        "COMBINED_3P_2S",
        "DO_IDO",
        "3P+2S",
        "-homlok",
        ("homlok",),
        ("homlok", "omlok", "womlok", "homlek", "omlek", "womlek"),
        282,
    ),
    SuffixSpec(
        "COMBINED_3P_3SM",
        "DO_IDO",
        "3P+3SM",
        "-homlu",
        ("homlu",),
        ("homlu", "omlu", "womlu"),
        283,
    ),
    SuffixSpec(
        "COMBINED_3P_3SF",
        "DO_IDO",
        "3P+3SF",
        "-homlha",
        ("homlha", "homlhie"),
        ("homlha", "omlha", "womlha", "homlhie", "omlhie", "womlhie", "homlie", "omlie", "womlie"),
        284,
    ),
    SuffixSpec(
        "COMBINED_3P_1P",
        "DO_IDO",
        "3P+1P",
        "-homlna",
        ("homlna",),
        ("homlna", "omlna", "womlna"),
        285,
    ),
    SuffixSpec(
        "COMBINED_3P_2P",
        "DO_IDO",
        "3P+2P",
        "-homlkom",
        ("homlkom",),
        ("homlkom", "omlkom", "womlkom"),
        286,
    ),
    SuffixSpec(
        "COMBINED_1P_3P",
        "DO_IDO",
        "1P+3P",
        "-nielhom",
        ("nielhom",),
        ("nielhom", "nilhom", "nielom", "nilom"),
        290,
    ),
    SuffixSpec(
        "COMBINED_1P_DO3P",
        "DO_IDO",
        "1P+3P",
        "-niehom",
        ("niehom",),
        ("niehom", "nijhom", "nijom", "niejom"),
        291,
    ),
    SuffixSpec(
        "COMBINED_1P_3P_3P",
        "DO_IDO",
        "1P+3P+3P",
        "-nihomlhom",
        ("nihomlhom",),
        (
            "nihomlhom",
            "nijhomlhom",
            "nijomlhom",
            "nihomlom",
            "nijhomlom",
            "nijomlom",
        ),
        292,
    ),
    SuffixSpec(
        "COMBINED_1P_3SM_3P",
        "DO_IDO",
        "1P+3SM+3P",
        "-niehulhom",
        ("niehulhom",),
        ("niehulhom", "nijulhom", "nijulom"),
        293,
    ),
    SuffixSpec(
        "COMBINED_1P_3SF_3P",
        "DO_IDO",
        "1P+3SF+3P",
        "-nihielhom",
        ("nihielhom",),
        ("nihielhom", "niehielhom", "nijhielhom", "nijielhom"),
        294,
    ),
    SuffixSpec(
        "COMBINED_3SF_1S",
        "DO_IDO",
        "3SF+1S",
        "-hieli",
        ("hieli",),
        ("hieli", "ieli"),
        310,
    ),
    SuffixSpec(
        "COMBINED_3SF_2S",
        "DO_IDO",
        "3SF+2S",
        "-hielek",
        ("hielek",),
        ("hielek", "ielek"),
        320,
    ),
    SuffixSpec(
        "COMBINED_3SF_3SM",
        "DO_IDO",
        "3SF+3SM",
        "-hielu",
        ("hielu",),
        ("hielu", "ielu"),
        330,
    ),
    SuffixSpec(
        "COMBINED_3SF_3SF",
        "DO_IDO",
        "3SF+3SF",
        "-hielha",
        ("hielha",),
        ("hielha", "ielha"),
        340,
    ),
    SuffixSpec(
        "COMBINED_3SF_1P",
        "DO_IDO",
        "3SF+1P",
        "-hielna",
        ("hielna",),
        ("hielna", "ielna", "ilna"),
        350,
    ),
    SuffixSpec(
        "COMBINED_3SF_2P",
        "DO_IDO",
        "3SF+2P",
        "-hielkom",
        ("hielkom",),
        ("hielkom", "ielkom"),
        360,
    ),
    SuffixSpec(
        "COMBINED_3SF_3P",
        "DO_IDO",
        "3SF+3P",
        "-hielhom",
        ("hielhom",),
        ("hielhom", "ielhom", "hielom", "ielom", "ilhom", "ilom"),
        370,
    ),
)

ALL_SUFFIXES: tuple[SuffixSpec, ...] = (
    *DO_SUFFIXES,
    *IDO_SUFFIXES,
    *COMBINED_SUFFIXES,
)


class MalteseSuffixRules:
    def __init__(
        self,
        *,
        spellchecker,
        verb_index: MalteseVerbFormIndex,
    ) -> None:
        self.spellchecker = spellchecker
        self.verb_index = verb_index

    # ------------------------------------------------------------------
    # Shared string helpers
    # ------------------------------------------------------------------

    def normalize(self, word: str) -> str:
        return self.spellchecker._normalize_word(word)

    def graphemes(self, word: str) -> list[str]:
        return self.spellchecker._graphemes(word)

    def from_graphemes(self, graphemes: Iterable[str]) -> str:
        return self.spellchecker._from_graphemes(graphemes)

    def is_vowel(self, grapheme: str) -> bool:
        return grapheme in self.spellchecker.VOWELS

    def is_consonant(self, grapheme: str) -> bool:
        return grapheme.isalpha() and not self.is_vowel(grapheme)

    def add_unique(
        self,
        items: list[tuple[str, str, str]],
        stem: str | None,
        rule_id: str,
        description: str,
    ) -> None:
        if not stem:
            return

        stem = self.normalize(stem)

        if not stem:
            return

        item = (stem, rule_id, description)

        if item not in items:
            items.append(item)

    # ------------------------------------------------------------------
    # Suffix parsing
    # ------------------------------------------------------------------

    def parse_suffixes(
        self,
        word: str,
        *,
        max_parses: int = 8,
    ) -> list[ParsedSuffix]:
        """
        Longest-ending-first suffix parser.

        Example:
            għamilkom
                first sees lkom/ilkom family,
                not every possible suffix.
        """
        normalized = self.normalize(word)
        parses: list[ParsedSuffix] = []

        ending_rows: list[tuple[str, SuffixSpec]] = []

        for spec in ALL_SUFFIXES:
            for ending in spec.typed_endings:
                ending_rows.append((ending, spec))

        ending_rows.sort(
            key=lambda item: (
                -len(item[0]),
                item[1].priority,
            )
        )

        seen: set[tuple[str, str]] = set()

        for ending, spec in ending_rows:
            if not normalized.endswith(ending):
                continue

            stem = normalized[:-len(ending)]

            if not stem:
                continue

            if spec.label == "DO_3SF" and ending == "a":
                stem_graphemes = self.graphemes(stem)
                if (
                    len(stem_graphemes) < 4
                    or stem_graphemes[-1] == "j"
                    or not self.is_consonant(stem_graphemes[-1])
                ):
                    continue

            key = (spec.label, ending)

            if key in seen:
                continue

            parses.append(
                ParsedSuffix(
                    spec=spec,
                    typed_ending=ending,
                    typed_stem=stem,
                    priority=spec.priority,
                )
            )
            seen.add(key)

            if len(parses) >= max_parses:
                break

        return parses

    # ------------------------------------------------------------------
    # Stem transformations
    # ------------------------------------------------------------------

    def ends_vowel_consonant(self, word: str) -> bool:
        g = self.graphemes(word)
        return (
            len(g) >= 2
            and self.is_vowel(g[-2])
            and self.is_consonant(g[-1])
        )

    def ends_e_consonant(self, word: str) -> bool:
        g = self.graphemes(word)
        return (
            len(g) >= 2
            and g[-2] == "e"
            and self.is_consonant(g[-1])
        )

    def final_eC_to_iC(self, word: str) -> str | None:
        g = self.graphemes(word)

        if (
            len(g) >= 2
            and g[-2] == "e"
            and self.is_consonant(g[-1])
        ):
            return self.from_graphemes(g[:-2] + ["i", g[-1]])

        return None

    def final_eC_drop_e(self, word: str) -> str | None:
        g = self.graphemes(word)

        if (
            len(g) >= 2
            and g[-2] == "e"
            and self.is_consonant(g[-1])
        ):
            return self.from_graphemes(g[:-2] + [g[-1]])

        return None

    def final_VC_drop_vowel(self, word: str) -> str | None:
        g = self.graphemes(word)

        if (
            len(g) >= 2
            and self.is_vowel(g[-2])
            and self.is_consonant(g[-1])
        ):
            return self.from_graphemes(g[:-2] + [g[-1]])

        return None

    def final_VC_to_iC(self, word: str) -> str | None:
        g = self.graphemes(word)

        if (
            len(g) >= 2
            and self.is_vowel(g[-2])
            and self.is_consonant(g[-1])
        ):
            return self.from_graphemes(g[:-2] + ["i", g[-1]])

        return None

    def final_doubled_consonant_add_i(self, word: str) -> str | None:
        g = self.graphemes(word)

        if (
            len(g) >= 2
            and g[-1] == g[-2]
            and self.is_consonant(g[-1])
        ):
            return self.from_graphemes(g + ["i"])

        return None

    def final_consonant_add_i(self, word: str) -> str | None:
        g = self.graphemes(word)

        if g and self.is_consonant(g[-1]):
            return self.from_graphemes(g + ["i"])

        return None

    def final_VCa_to_Cie(self, word: str) -> str | None:
        """
        nesa -> nsie
        beda -> bdie
        """
        g = self.graphemes(word)

        if (
            len(g) >= 3
            and self.is_vowel(g[-3])
            and self.is_consonant(g[-2])
            and g[-1] == "a"
        ):
            return self.from_graphemes(g[:-3] + [g[-2], "i", "e"])

        return None

    def final_a_to_ie(self, word: str) -> str | None:
        g = self.graphemes(word)

        if g and g[-1] == "a":
            return self.from_graphemes(g[:-1] + ["i", "e"])

        return None

    def imp_final_a_to_i_variants(self, word: str) -> list[str]:
        """
        insa -> insi
        insa -> nsi

        Both are generated because the rule document has imperative examples
        where the initial i may effectively disappear.
        """
        g = self.graphemes(word)
        variants: list[str] = []

        if not g or g[-1] != "a":
            return variants

        variants.append(self.from_graphemes(g[:-1] + ["i"]))

        if len(g) >= 2 and g[0] == "i":
            variants.append(self.from_graphemes(g[1:-1] + ["i"]))

        return variants

    def imp_final_a_to_ghu(self, word: str) -> str | None:
        """
        itfa -> itfgħu before plural direct-object suffixes.
        """
        g = self.graphemes(word)

        if not g or g[-1] != "a":
            return None

        return self.from_graphemes(g[:-1] + ["għ", "u"])

    def ieCeC_to_eCiC(self, word: str) -> str | None:
        """
        stieden -> stedin
        bierek  -> berik
        """
        g = self.graphemes(word)

        if len(g) >= 5:
            a, b, c, d, e = g[-5:]

            if (
                a == "i"
                and b == "e"
                and self.is_consonant(c)
                and d == "e"
                and self.is_consonant(e)
            ):
                    return self.from_graphemes(
                        g[:-5] + ["e", c, "i", e]
                    )

        return None

    def ie_only_in_last_two_syllables(self, word: str) -> str | None:
        """
        Keep ie only in the last two syllables.

        If ie appears earlier than that, reduce it to i.
        """
        g = self.graphemes(word)
        nuclei: list[tuple[int, int, str]] = []
        i = 0

        while i < len(g):
            if i + 1 < len(g) and g[i] == "i" and g[i + 1] == "e":
                nuclei.append((i, 2, "ie"))
                i += 2
                continue

            if self.is_vowel(g[i]):
                nuclei.append((i, 1, g[i]))

            i += 1

        if len(nuclei) <= 2:
            return None

        offending_starts: set[int] = set()
        for index, (start, length, kind) in enumerate(nuclei):
            if kind == "ie" and (len(nuclei) - index - 1) >= 2:
                offending_starts.add(start)

        if not offending_starts:
            return None

        out: list[str] = []
        i = 0

        while i < len(g):
            if i in offending_starts:
                out.append("i")
                i += 2
                continue

            out.append(g[i])
            i += 1

        return self.from_graphemes(out)

    def perf_3p_wie_to_wi(self, word: str) -> str | None:
        """
        wieġbu -> wiġbu before object suffixes.
        """
        g = self.graphemes(word)

        if len(g) >= 4 and g[:3] == ["w", "i", "e"]:
            return self.from_graphemes(["w", "i", *g[3:]])

        return None

    def final_apostrophe_to_gh(self, word: str) -> str | None:
        g = self.graphemes(word)

        if g and g[-1] == "'":
            return self.from_graphemes(g[:-1] + ["għ"])

        return None

    def final_apostrophe_to_gha(self, word: str) -> str | None:
        g = self.graphemes(word)

        if g and g[-1] == "'":
            return self.from_graphemes(g[:-1] + ["għ", "a"])

        return None

    def final_a_apostrophe_to_gh(self, word: str) -> str | None:
        """
        bela' + -u -> belgħu
        """
        g = self.graphemes(word)

        if (
            len(g) >= 2
            and g[-2] == "a"
            and g[-1] == "'"
        ):
            return self.from_graphemes(g[:-2] + ["għ"])

        return None

    def perf_1p_drop_final_a(self, word: str) -> str | None:
        g = self.graphemes(word)

        if len(g) >= 2 and g[-2:] == ["n", "a"]:
            return self.from_graphemes(g[:-1])

        return None

    def f1_medial_vccvc_to_vcvcc(
        self,
        word: str,
        *,
        add_i: bool = False,
    ) -> str | None:
        """
        F1 with R2 = l/m/n/r/għ:
            VCCVC -> VCVCC
            VCCVC -> VCVCCi

        Example:
            nisraq -> nisirq
            nisraq -> nisirqi
        """
        g = self.graphemes(word)

        if len(g) >= 5:
            v1, c1, c2, v2, c3 = g[-5:]

            if (
                self.is_vowel(v1)
                and self.is_consonant(c1)
                and self.is_consonant(c2)
                and self.is_vowel(v2)
                and self.is_consonant(c3)
            ):
                out = g[:-5] + [v1, c1, v1, c2, c3]

                if add_i:
                    out.append("i")

                return self.from_graphemes(out)

        return None

    # ------------------------------------------------------------------
    # Forward generation by suffix family
    # ------------------------------------------------------------------

    def surface_suffixes_for_record(
        self,
        record: VerbFormRecord,
        spec: SuffixSpec,
    ) -> tuple[str, ...]:
        base = record.word
        g = self.graphemes(base)

        if spec.label == "DO_2S":
            if g and self.is_vowel(g[-1]):
                return ("k",)

            if "o" in [token for token in g[-4:]]:
                return ("ok", "ek")

            return ("ek", "k")

        if spec.label == "DO_3SM":
            if g and self.is_vowel(g[-1]):
                return ("h",)

            return ("u",)

        if spec.label == "COMBINED_1P_DO3P":
            root_class = self.verb_index.root_class(record)
            if (
                record.is_perf
                and record.person == "1P"
                and root_class.startswith("F1_")
                and not record.root.startswith("għ")
            ):
                return ("ihom",)

            if record.is_perf and record.person == "1P":
                return ("iehom",)

        if spec.label == "COMBINED_1P_3P_3P":
            if record.is_perf and record.person == "1P":
                return ("ihomlhom",)

        return spec.canonical_surfaces

    def stems_for_spec(
        self,
        record: VerbFormRecord,
        spec: SuffixSpec,
    ) -> list[tuple[str, str, str]]:
        base = record.word
        stems: list[tuple[str, str, str]] = []

        root_class = self.verb_index.root_class(record)
        is_medial = root_class == "F1_MEDIAL_LIQUID_OR_GUTTURAL"

        # Default addition is always a possible fallback.
        self.add_unique(
            stems,
            base,
            "DEFAULT_ADD",
            "add suffix directly",
        )

        if record.is_stem and record.stem_kind == "IS":
            self.add_unique(
                stems,
                self.final_a_to_ie(base),
                "IS_FINAL_A_TO_IE",
                "nonsemitic IS final -a -> -ie before suffix",
            )

        # Special F1 medial liquid/guttural rules.
        if is_medial and spec.kind in {"DO", "IDO", "DO_IDO"}:
            add_i = spec.kind in {"IDO", "DO_IDO"} and spec.label not in {
                "IDO_1S",
                "IDO_2S",
                "IDO_3SM",
            }

            medial_stem = self.f1_medial_vccvc_to_vcvcc(
                base,
                add_i=add_i,
            )

            self.add_unique(
                stems,
                medial_stem,
                "F1_MEDIAL_VCCVC",
                (
                    "F1 R2=l/m/n/r/għ VCCVC -> VCVCC"
                    + ("i" if add_i else "")
                ),
            )

        # Final weak j/w behaviour.
        if self.verb_index.is_final_weak(record):
            if record.is_imp:
                for variant in self.imp_final_a_to_i_variants(base):
                    self.add_unique(
                        stems,
                        variant,
                        "IMP_A_TO_I",
                        "imperative final -a -> -i",
                    )
                if spec.label == "DO_3P":
                    self.add_unique(
                        stems,
                        self.imp_final_a_to_ghu(base),
                        "IMP_A_TO_GHU",
                        "imperative final -a -> -għu before -hom",
                    )
            else:
                self.add_unique(
                    stems,
                    self.final_VCa_to_Cie(base),
                    "FINAL_WEAK_VCA_TO_CIE",
                    "final weak VCa -> Cie",
                )

        # ieCeC pattern such as stieden/bierek.
        self.add_unique(
            stems,
            self.ieCeC_to_eCiC(base),
            "IECEC_TO_ECIC",
            "ieCeC -> eCiC",
        )

        # Apostrophe rules.  Multiple variants are kept intentionally because
        # the rule document gives different outputs depending on suffix type.
        if base.endswith("'"):
            self.add_unique(
                stems,
                self.final_apostrophe_to_gh(base),
                "FINAL_APOSTROPHE_TO_GH",
                "final apostrophe -> għ",
            )

            if spec.kind in {"IDO", "DO_IDO"}:
                self.add_unique(
                    stems,
                    self.final_apostrophe_to_gha(base),
                    "FINAL_APOSTROPHE_TO_GHA",
                    "final apostrophe -> għa before l-suffix",
                )

            if spec.label == "DO_3SM":
                self.add_unique(
                    stems,
                    self.final_a_apostrophe_to_gh(base),
                    "FINAL_A_APOSTROPHE_TO_GH",
                    "final a' -> għ before -u",
                )

        # DO-specific normal rules.
        if spec.kind == "DO":
            if record.is_perf and record.person == "3P":
                self.add_unique(
                    stems,
                    self.perf_3p_wie_to_wi(base),
                    "PERF_3P_WIE_TO_WI",
                    "perfect 3P wie- -> wi- before object suffix",
                )

            if spec.label == "DO_1S":
                self.add_unique(
                    stems,
                    self.final_eC_to_iC(base),
                    "DO_NI_EC_TO_IC",
                    "eC -> iC before -ni",
                )
                self.add_unique(
                    stems,
                    self.final_VC_to_iC(base),
                    "DO_NI_VC_TO_IC",
                    "VC -> iC before -ni",
                )

            elif spec.label == "DO_2S":
                self.add_unique(
                    stems,
                    self.final_eC_drop_e(base),
                    "DO_2S_EC_DROP_E",
                    "eC -> C before -ek/-ok/-k",
                )
                self.add_unique(
                    stems,
                    self.final_VC_drop_vowel(base),
                    "DO_2S_VC_DROP_V",
                    "VC -> C before -ek/-ok/-k",
                )

            elif spec.label == "DO_3SM":
                self.add_unique(
                    stems,
                    self.final_eC_drop_e(base),
                    "DO_3SM_EC_DROP_E",
                    "eC -> C before -u",
                )
                self.add_unique(
                    stems,
                    self.final_VC_drop_vowel(base),
                    "DO_3SM_VC_DROP_V",
                    "VC -> C before -u",
                )

            elif spec.label in {"DO_3SF", "DO_1P", "DO_2P", "DO_3P"}:
                self.add_unique(
                    stems,
                    self.final_eC_to_iC(base),
                    "DO_PL_EC_TO_IC",
                    "eC -> iC before -ha/-na/-kom/-hom",
                )

        # IDO / combined rules.
        if spec.kind in {"IDO", "DO_IDO"}:
            if (
                record.is_perf
                and record.person == "1P"
                and (
                    spec.label == "COMBINED_1P_3P_3P"
                    or (
                        spec.label == "COMBINED_1P_DO3P"
                        and root_class.startswith("F1_")
                    )
                )
            ):
                self.add_unique(
                    stems,
                    self.perf_1p_drop_final_a(base),
                    "PERF_1P_DROP_FINAL_A",
                    "perfect 1P final -a drops before object suffix",
                )

            if spec.kind == "DO_IDO" and spec.person.startswith("3SM+"):
                self.add_unique(
                    stems,
                    self.final_eC_to_iC(base),
                    "COMBINED_3SM_EC_TO_IC",
                    "eC -> iC before -hu- combined suffix",
                )
                self.add_unique(
                    stems,
                    self.final_VC_to_iC(base),
                    "COMBINED_3SM_VC_TO_IC",
                    "VC -> iC before -hu- combined suffix",
                )

            if spec.kind == "DO_IDO" and spec.person.startswith("3SF+"):
                self.add_unique(
                    stems,
                    self.final_eC_to_iC(base),
                    "COMBINED_3SF_EC_TO_IC",
                    "eC -> iC before -hie- combined suffix",
                )
                self.add_unique(
                    stems,
                    self.final_VC_to_iC(base),
                    "COMBINED_3SF_VC_TO_IC",
                    "VC -> iC before -hie- combined suffix",
                )

            if spec.label == "COMBINED_1P_3P":
                self.add_unique(
                    stems,
                    self.final_eC_to_iC(base),
                    "COMBINED_1P_EC_TO_IC",
                    "eC -> iC before -nie- combined suffix",
                )
                self.add_unique(
                    stems,
                    self.final_VC_to_iC(base),
                    "COMBINED_1P_VC_TO_IC",
                    "VC -> iC before -nie- combined suffix",
                )

            if spec.label in {
                "COMBINED_1P_DO3P",
                "COMBINED_1P_3P_3P",
                "COMBINED_1P_3SM_3P",
                "COMBINED_1P_3SF_3P",
            }:
                self.add_unique(
                    stems,
                    self.final_eC_to_iC(base),
                    "COMBINED_1P_DO_EC_TO_IC",
                    "eC -> iC before -nie- direct-object suffix",
                )
                self.add_unique(
                    stems,
                    self.final_VC_to_iC(base),
                    "COMBINED_1P_DO_VC_TO_IC",
                    "VC -> iC before -nie- direct-object suffix",
                )

            if spec.label.startswith("COMBINED_3P_"):
                self.add_unique(
                    stems,
                    self.final_eC_to_iC(base),
                    "COMBINED_3P_EC_TO_IC",
                    "eC -> iC before -homl- combined suffix",
                )

            if spec.label in {"IDO_1S", "IDO_2S", "IDO_3SM"}:
                self.add_unique(
                    stems,
                    self.final_eC_to_iC(base),
                    "IDO_SHORT_EC_TO_IC",
                    "eC -> iC before -li/-lek/-lu",
                )
            elif spec.kind == "IDO":
                # lha/lna/lkom/lhom and combined huli/hielkom-style endings.
                # This gives:
                #   għamel -> għamli + lhom = għamlilhom
                #   kiser  -> kisri + lha  = kisrilha
                self.add_unique(
                    stems,
                    self.final_doubled_consonant_add_i(base),
                    "IDO_L_FORMS_DOUBLE_C_ADD_I",
                    "doubled final consonant -> CC+i before l-form",
                )

                if root_class == "HOLLOW":
                    self.add_unique(
                        stems,
                        self.final_consonant_add_i(base),
                        "IDO_L_FORMS_HOLLOW_ADD_I",
                        "hollow verb consonant-final base -> base+i before l-form",
                    )

                dropped = self.final_eC_drop_e(base)

                if dropped:
                    self.add_unique(
                        stems,
                        dropped + "i",
                        "IDO_L_FORMS_EC_TO_CI",
                        "eC -> C+i before l-form",
                    )

                dropped_vc = self.final_VC_drop_vowel(base)

                if dropped_vc:
                    self.add_unique(
                        stems,
                        dropped_vc + "i",
                        "IDO_L_FORMS_VC_TO_CI",
                        "VC -> C+i before l-form",
                    )

        return stems

    def generate_for_record_and_spec(
        self,
        record: VerbFormRecord,
        spec: SuffixSpec,
    ) -> list[GeneratedSuffixCandidate]:
        if (
            spec.kind == "DO"
            and self.verb_index.is_blocked_for_direct_object(record)
        ):
            return []

        candidates: list[GeneratedSuffixCandidate] = []
        seen: set[str] = set()
        root_class = self.verb_index.root_class(record)

        for stem, rule_id, description in self.stems_for_spec(record, spec):
            for suffix in self.surface_suffixes_for_record(record, spec):
                surface = self.normalize(stem + suffix)

                if record.is_stem and record.stem_kind == "IS":
                    surface = self.ie_only_in_last_two_syllables(surface) or surface

                if not surface or surface in seen:
                    continue

                seen.add(surface)

                candidates.append(
                    GeneratedSuffixCandidate(
                        surface=surface,
                        base=record.word,
                        suffix_label=spec.label,
                        suffix_kind=spec.kind,
                        suffix_person=spec.person,
                        suffix_display=spec.display,
                        rule_id=rule_id,
                        rule_description=description,
                        raw_tag=record.raw_tag,
                        root=record.root,
                        form_class=record.form_class,
                        tense=record.tense,
                        person=record.person,
                        root_class=root_class,
                    )
                )

        return candidates
