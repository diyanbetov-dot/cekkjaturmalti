from typing import List, Optional
from spellchecker.schema import Candidate, ErrorClass, RiskClass, Token
from spellchecker.config import MAX_TOKEN_CANDIDATES, MAX_SPAN_CANDIDATES
from .spans import deduplicate_candidates
from .orthography import generate_orthographic_candidates
from .initial_i import generate_initial_i_candidates
from .articles import generate_article_span_candidates
from .numerals import generate_numeral_span_candidates
from .error_memory import generate_error_memory_candidates
from .morphology import generate_agreement_span_candidates


class CandidateGenerator:
    def __init__(self, lexicon, english_lexicon, entity_lexicon) -> None:
        self.lexicon = lexicon
        self.english_lexicon = english_lexicon
        self.entity_lexicon = entity_lexicon

    def generate_candidates(self, tokens: List[Token]) -> List[Candidate]:
        all_candidates: List[Candidate] = []

        for i, token in enumerate(tokens):
            if token.token_type != "word":
                continue

            # 1. KEEP Candidate (Mandatory)
            keep_cand = Candidate(
                source_start=token.start,
                source_end=token.end,
                original_text=token.text,
                replacement=token.text,
                operation_type=ErrorClass.KEEP,
                risk_class=RiskClass.LOW,
                sources=["KEEP"],
                hard_valid=True,
            )
            all_candidates.append(keep_cand)

            # Check if token is protected English or Entity
            if self.english_lexicon.is_english(token.normalized) or self.entity_lexicon.is_entity(token.normalized):
                casing_target = self.entity_lexicon.get_casing_candidate(token.text)
                if casing_target != token.text:
                    all_candidates.append(
                        Candidate(
                            source_start=token.start,
                            source_end=token.end,
                            original_text=token.text,
                            replacement=casing_target,
                            operation_type=ErrorClass.CAPITALIZATION,
                            risk_class=RiskClass.LOW,
                            sources=["entity_casing"],
                            hard_valid=True,
                        )
                    )
                continue

            # 2. Orthographic Candidates
            orth_cands = generate_orthographic_candidates(token, self.lexicon)
            all_candidates.extend(orth_cands)

            # 3. Initial-i Candidates
            init_i_cands = generate_initial_i_candidates(tokens, i, self.lexicon)
            all_candidates.extend(init_i_cands)

            # 4. Error Memory Candidates
            mem_cands = generate_error_memory_candidates(token)
            all_candidates.extend(mem_cands)

            # 5. Span Candidates (Articles / Prepositions / Numerals / Agreement)
            art_span_cands = generate_article_span_candidates(tokens, i, self.lexicon)
            all_candidates.extend(art_span_cands)

            num_span_cands = generate_numeral_span_candidates(tokens, i, self.lexicon)
            all_candidates.extend(num_span_cands)

            agreed_cands = generate_agreement_span_candidates(tokens, i, self.lexicon)
            all_candidates.extend(agreed_cands)

        # Deduplicate and cap
        return deduplicate_candidates(all_candidates)
