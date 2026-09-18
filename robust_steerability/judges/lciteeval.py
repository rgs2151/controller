"""Released L-CiteEval citation scoring logic, independent of generation."""

from __future__ import annotations

import copy
import re
from collections.abc import Callable

from robust_steerability.judges.exact import remove_citations
from robust_steerability.judges.specs import scorer_spec


EntailmentFunction = Callable[[str, str], bool]


def _format_document(document: str | dict) -> str:
    if isinstance(document, str):
        return f"Passage: {document}"
    return f"Title: {document['title']}\nPassage: {document['text']}"


def _sentences(text: str) -> list[str]:
    """Split generated claims with the same NLTK tokenizer as L-CiteEval."""

    from nltk import sent_tokenize

    return sent_tokenize(text)


def load_lcite_entailer(device: int):
    """Load the exact pinned local NLI backend used for citation scoring."""

    from transformers import pipeline

    specification = scorer_spec("lcite_citation_nli")
    return pipeline(
        "text-classification",
        model=specification.model_id,
        revision=specification.revision,
        device=device,
    )


def pipeline_entailment(pipeline_object) -> EntailmentFunction:
    """Adapt a Transformers NLI pipeline to the citation scorer interface."""

    def entails(passage: str, claim: str) -> bool:
        result = pipeline_object([{"text": passage, "text_pair": claim}])[0]
        return str(result["label"]).lower() == "entailment"

    return entails


def lcite_citation_scores(
    completion: str,
    documents: list[str | dict],
    entails: EntailmentFunction,
    *,
    at_most_citations: int = 3,
) -> dict[str, float | int]:
    """Compute L-CiteEval AutoAIS citation precision, recall, and F1."""

    sentences = _sentences(completion)
    if not sentences:
        return {
            "citation_precision": 0.0,
            "citation_recall": 0.0,
            "citation_f1": 0.0,
            "citation_count": 0,
        }
    entailed_claims = 0
    necessary_citations = 0
    citation_count = 0
    for sentence in sentences:
        claim = remove_citations(sentence).strip()
        references = [int(value) - 1 for value in re.findall(r"\[(\d+)", sentence)]
        joint_entailment = False
        valid = bool(references) and all(
            0 <= reference < len(documents) for reference in references
        )
        if valid:
            references = references[:at_most_citations]
            citation_count += len(references)
            joint_passage = "\n".join(
                _format_document(documents[reference]) for reference in references
            )
            joint_entailment = bool(entails(joint_passage, claim))
        entailed_claims += int(joint_entailment)
        if not joint_entailment:
            continue
        if len(references) == 1:
            necessary_citations += 1
            continue
        for reference in references:
            passage = _format_document(documents[reference])
            if entails(passage, claim):
                necessary_citations += 1
                continue
            remainder = copy.copy(references)
            remainder.remove(reference)
            without_reference = "\n".join(
                _format_document(documents[index]) for index in remainder
            )
            if not entails(without_reference, claim):
                necessary_citations += 1
    recall = entailed_claims / len(sentences)
    precision = necessary_citations / citation_count if citation_count else 0.0
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    return {
        "citation_precision": precision,
        "citation_recall": recall,
        "citation_f1": f1,
        "citation_count": citation_count,
    }
