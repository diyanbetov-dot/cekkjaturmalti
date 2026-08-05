from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from neural_corrector.inference.corrector import NeuralCorrector
from neural_corrector.models.alignment import COPY_ACTION, derive_actions

from contextual_corrector import (
    CandidateGenerationPipeline,
    CandidateLattice,
    CandidateOperation,
    DictionaryCandidateAdapter,
    LatticeLimits,
    NeuralCandidateAdapter,
    PhraseOrthographicCandidateAdapter,
    SourceEvidence,
    SpanCandidate,
    Stage1CandidateAdapter,
    SuffixCandidateAdapter,
    apply_candidate_path,
    normalize_for_lattice,
    sentence_id_for_text,
)


def suffix_row(surface: str, kind: str = "DO", person: str = "3SM"):
    return SimpleNamespace(
        surface=surface,
        base="tefgħu",
        suffix_label=f"{kind}_{person}",
        suffix_kind=kind,
        suffix_person=person,
        suffix_display="-h" if person == "3SM" else "",
        rule_id="fake_suffix",
        rule_description="test",
        raw_tag="T-w-tfg-F1-IMP-2P",
        root="tfgħ",
        form_class="F1",
        tense="IMP",
        person="2P",
        root_class="sound",
    )


class FakeVerbIndex:
    def word_records(self, word):
        if word == "tefgħu":
            return [SimpleNamespace(word="tefgħu", root="tfgħ", form_class="F1", tense="IMP", person="2P")]
        return []


class FakeOrthographic:
    def dictionary_shortcut_variants(self, word):
        return {"hareg": ["ħareġ"], "Censu": ["Ċensu"]}.get(word, [])

    def dictionary_gh_priority_variants(self, word):
        return []

    def dictionary_final_aw_to_ghu_variants(self, word):
        return []

    def dictionary_i_ie_variants(self, word):
        return []

    def dictionary_d_t_variants(self, word):
        return []

    def dictionary_b_p_variants(self, word):
        return []

    def dictionary_g_k_cluster_variants(self, word):
        return []


class FakeSuffixGenerator:
    def __init__(self, spellchecker):
        self.spellchecker = spellchecker
        self.verb_index = FakeVerbIndex()

    def exact_suffix_matches(self, word):
        return [suffix_row(word)] if word == "iwassalhom" else []

    def parse_possible_suffixes(self, word):
        return []

    def suggest_suffixes(self, word, limit=4):
        return []

    def candidates_for_surface(self, word, limit=16):
        return [suffix_row("tefgħuh")] if word == "tefgħuh" else []


class FakeSpellchecker:
    def __init__(self):
        self.dictionary_set = {"ċensu", "ħareġ", "minn", "tefgħu", "barra", "għax", "ma", "riedx", "iwassalhom"}
        self.word_tags = {
            "ħareġ": {"T-ħ-r-ġ-F1-PERF-3SM"},
            "tefgħu": {"T-t-f-għ-F1-IMP-2P"},
        }
        self.orthographic_generator = FakeOrthographic()
        self.suffix_generator = FakeSuffixGenerator(self)
        self.calls = []

    @staticmethod
    def _normalize_word(word):
        return word.casefold()

    def _symspell_candidates(self, word, limit=2):
        return ["ħareġ", "tefgħu", "riedx"][:limit]

    def correct_word(self, word):
        return {"andek": "għandek", "ghamilt": "għamilt"}.get(word.casefold(), word)

    def correct_text_rich(self, text):
        self.calls.append(text)
        corrected = text.replace("Censu", "Ċensu").replace("hareg", "ħareġ")
        tokens = []
        for original, replacement in (("Censu", "Ċensu"), ("hareg", "ħareġ")):
            if original in text:
                tokens.append({
                    "type": "word", "original": original, "corrected": replacement,
                    "choices": [{"word": replacement}], "ambiguous": False,
                    "recognition_sources": ["dictionary"], "unrecognized": False,
                })
        return {"corrected_text": corrected, "tokens": tokens}


def lattice(text: str, *, limits=None):
    raw = normalize_for_lattice(text)
    return CandidateLattice(
        sentence_id=sentence_id_for_text(raw.normalized), raw=raw, limits=limits
    )


def test_stage1_baseline_path_is_reconstructable_and_api_is_preserved():
    spellchecker = FakeSpellchecker()
    raw = "Censu hareg."
    result = Stage1CandidateAdapter(spellchecker).generate_candidates(raw)
    assert callable(spellchecker.correct_word)
    assert callable(spellchecker.correct_text_rich)
    baseline = [candidate for candidate in result.candidates if candidate.candidate_id in result.baseline_candidate_ids]
    assert apply_candidate_path(raw, baseline) == result.s1_text


def test_pipeline_runs_stage1_once_and_does_not_return_a_selected_sentence():
    spellchecker = FakeSpellchecker()
    result = CandidateGenerationPipeline(
        spellchecker=spellchecker, include_fuzzy=False
    ).generate_candidate_lattice("Censu hareg.")
    assert spellchecker.calls == ["Censu hareg."]
    assert not hasattr(result, "corrected_text")
    assert result.production_stage1_result["corrected_text"] == result.s1_text


def test_complete_keep_path_covers_words_punctuation_and_newlines():
    spellchecker = FakeSpellchecker()
    result = CandidateGenerationPipeline(
        spellchecker=spellchecker, include_fuzzy=False
    ).generate_candidate_lattice("Censu,\n\n hareg!")
    keeps = [edge for edge in result.lattice.edges if edge.keep]
    assert len(keeps) == len(result.tokens)
    assert result.lattice.has_complete_keep_path()
    assert apply_candidate_path(result.raw_text, keeps) == result.raw_text


def test_dictionary_existence_is_evidence_not_a_lock():
    spellchecker = FakeSpellchecker()
    graph = lattice("hareg")
    proposals = DictionaryCandidateAdapter(spellchecker).generate_candidates("hareg", graph)
    for candidate in proposals:
        graph.add(candidate)
    edges = graph.finalize()
    assert any("fuzzy" in edge.sources for edge in edges)
    assert all(not edge.hard_violations for edge in edges)


def test_fuzzy_adapter_emits_at_most_two_candidates_per_span():
    graph = lattice("hareg")
    proposals = DictionaryCandidateAdapter(FakeSpellchecker()).generate_candidates("hareg", graph)
    assert len([candidate for candidate in proposals if "fuzzy" in candidate.sources]) <= 2


def test_phrase_adapter_supports_split_and_merge_spans():
    adapter = PhraseOrthographicCandidateAdapter(FakeSpellchecker())
    split_graph = lattice("xandek")
    split = adapter.generate_candidates("xandek", split_graph)
    assert any(candidate.replacement == "x'għandek" and candidate.operation == CandidateOperation.SPLIT for candidate in split)
    merge_graph = lattice("ma hawnx")
    merged = adapter.generate_candidates("ma hawnx", merge_graph)
    assert any(candidate.replacement == "m'hawnx" and candidate.operation == CandidateOperation.MERGE for candidate in merged)


def test_required_phrase_and_orthographic_examples_are_exposed():
    adapter = PhraseOrthographicCandidateAdapter(FakeSpellchecker())
    examples = {
        "mghamilt": "ma għamilt",
        "il lejla": "illejla",
        "illejla tal-festa": "il-lejla tal-festa",
        "daqs li kieku": "daqslikieku",
    }
    for source, expected in examples.items():
        graph = lattice(source)
        proposals = adapter.generate_candidates(source, graph)
        assert any(candidate.replacement == expected for candidate in proposals)


def test_contextual_illejla_forms_remain_competing_paths():
    graph = lattice("illejla tal-festa")
    proposals = PhraseOrthographicCandidateAdapter().generate_candidates(
        graph.raw.normalized, graph
    )
    graph.finalize()
    assert graph.has_complete_keep_path()
    assert any(candidate.replacement == "il-lejla tal-festa" for candidate in proposals)


def test_phrase_offsets_exclude_adjacent_punctuation():
    graph = lattice("(ma hawnx),")
    proposals = PhraseOrthographicCandidateAdapter().generate_candidates("(ma hawnx),", graph)
    candidate = next(value for value in proposals if value.replacement == "m'hawnx")
    assert candidate.raw_span.text == "ma hawnx"
    assert graph.raw.normalized[candidate.raw_span.char_start:candidate.raw_span.char_end] == "ma hawnx"


def test_suffix_adapter_distinguishes_unsuffixed_and_introduced_object():
    spellchecker = FakeSpellchecker()
    graph = lattice("tefghaw")
    proposals = SuffixCandidateAdapter(spellchecker.suffix_generator).generate_candidates(
        "tefghaw", graph.tokens, lattice=graph
    )
    unsuffixed = next(candidate for candidate in proposals if candidate.replacement == "tefgħu")
    introduced = next(candidate for candidate in proposals if candidate.replacement == "tefgħuh")
    assert all(not analysis.direct_object for analysis in unsuffixed.suffix_evidence)
    assert not unsuffixed.unsupported_clitic_insertion
    assert tuple(feature.label for feature in introduced.introduced_features) == ("DO:3SM",)
    assert introduced.unsupported_clitic_insertion
    assert all(not feature.input_evidence for feature in introduced.introduced_features)


def test_suffix_generation_is_limited_to_four_per_span():
    graph = lattice("iwassalhom", limits=LatticeLimits(suffix_candidates_per_span=4))
    span = graph.span(0, 1)
    for index in range(7):
        graph.add(graph.make_candidate(
            span=span, replacement=f"iwassal{index}", operation=CandidateOperation.REPLACE,
            sources={"suffix": (SourceEvidence(source="suffix", rank=index, calibrated=False),)},
            suffix_evidence=(),
        ))
    assert len([edge for edge in graph.finalize() if "suffix" in edge.sources]) <= 4


def test_duplicate_sources_merge_without_confidence_aggregation():
    graph = lattice("hareg")
    span = graph.span(0, 1)
    for source, score in (("stage1", 0.7), ("bigru", 0.9), ("bigru", 0.6)):
        graph.add(graph.make_candidate(
            span=span, replacement="ħareġ", operation=CandidateOperation.REPLACE,
            sources={source: (SourceEvidence(source=source, raw_score=score, source_confidence=score, calibrated=False),)},
        ))
    edge = next(candidate for candidate in graph.finalize() if candidate.replacement == "ħareġ")
    assert set(edge.sources) == {"stage1", "bigru"}
    assert [record.raw_score for record in edge.sources["bigru"]] == [0.9, 0.6]
    assert all(not record.calibrated for record in edge.evidence_records())


def test_stage1_and_neural_sources_do_not_suppress_each_other():
    graph = lattice("hareg")
    span = graph.span(0, 1)
    for replacement, source in (("ħareġ", "stage1"), ("ħarġet", "bigru")):
        graph.add(graph.make_candidate(
            span=span, replacement=replacement, operation=CandidateOperation.REPLACE,
            sources={source: (SourceEvidence(source=source, calibrated=False),)},
        ))
    replacements = {edge.replacement for edge in graph.finalize()}
    assert {"ħareġ", "ħarġet"}.issubset(replacements)


def test_bigru_candidate_api_is_prevalidation_and_capped_for_recognized_word():
    corrector = NeuralCorrector.__new__(NeuralCorrector)
    corrector.max_length = 64
    corrector.model_version = "test"

    def predict(chunk):
        target_actions = derive_actions(chunk, "ħareġ")
        rows = []
        for action in target_actions:
            alternatives = [(action, 0.91)]
            if action != COPY_ACTION:
                alternatives.append((COPY_ACTION, 0.08))
            alternatives.append(("", 0.01))
            rows.append(alternatives)
        return target_actions, [0.91] * len(chunk), rows

    corrector._predict_chunk = predict
    proposals = NeuralCandidateAdapter(corrector).generate_candidates("hareg", top_k=3)
    assert any(candidate.replacement == "ħareġ" for candidate in proposals)
    assert all("bigru" in candidate.sources for candidate in proposals)
    assert all(not record.calibrated for candidate in proposals for record in candidate.evidence_records())
    spans = {}
    for candidate in proposals:
        key = (candidate.raw_span.char_start, candidate.raw_span.char_end)
        spans[key] = spans.get(key, 0) + 1
    assert max(spans.values(), default=0) <= 3


def test_adapters_do_not_mutate_raw_text():
    raw = "Censu  hareg!"
    spellchecker = FakeSpellchecker()
    graph = lattice(raw)
    before = graph.raw
    Stage1CandidateAdapter(spellchecker).generate_candidates(raw)
    DictionaryCandidateAdapter(spellchecker).generate_candidates(raw, graph)
    PhraseOrthographicCandidateAdapter(spellchecker).generate_candidates(raw, graph)
    SuffixCandidateAdapter(spellchecker.suffix_generator).generate_candidates(raw, graph.tokens, lattice=graph)
    assert graph.raw == before
    assert raw == "Censu  hareg!"


def test_no_candidate_contains_a_final_selection_flag():
    graph = lattice("hareg")
    candidate = graph.make_candidate(
        span=graph.span(0, 1), replacement="ħareġ", operation=CandidateOperation.REPLACE,
        sources={"stage1": (SourceEvidence(source="stage1", calibrated=False),)},
    )
    assert not hasattr(candidate, "selected")
    assert candidate.ranker_score is None
