import pytest

from robust_steerability.benchmarks.metrics import (
    distinct_ngrams, judge_label, toxicity_frequency, truth_judge_prompt,
)
from robust_steerability.benchmarks.truthfulness import parse_mmlu_letter, _format_mmlu_question


def test_toxicity_is_label_frequency_not_confidence():
    rate, se = toxicity_frequency([0.49, 0.49, 0.51, 0.5])
    assert rate == 25
    assert se == pytest.approx(100 * (0.25 * 0.75 / 4) ** 0.5)
    with pytest.raises(ValueError):
        toxicity_frequency([float("nan")])


def test_distinct_matches_reference_corpus_boundaries():
    assert distinct_ngrams(["a b", "a b"]) == pytest.approx(2 / 3)
    assert distinct_ngrams(["a", "b"]) == 1
    assert distinct_ngrams([""]) == 0


@pytest.mark.parametrize("text,expected", [(" A", 0), ("B\n", 1), ("d", 3),
                                           ("", None), ("Answer", None), ("A or B", None),
                                           ("because", None), ("[A]", None)])
def test_mmlu_accepts_only_one_letter(text, expected):
    assert parse_mmlu_letter(text) == expected


def test_mmlu_reference_template():
    row = {"question": "Which?", "choices": ["one", "two", "three", "four"], "answer": 2}
    assert _format_mmlu_question(row, False) == "Question: Which?\nA. one\nB. two\nC. three\nD. four\nAnswer:"
    assert _format_mmlu_question(row, True).endswith("Answer: C")


def test_judges_receive_empty_answer_unchanged_and_parse_exact_labels():
    assert truth_judge_prompt("Question?", "", "True") == "Q: Question?\nA: \nTrue:"
    assert truth_judge_prompt("Question?", "I have no comment.", "Helpful").endswith("\nHelpful:")
    assert judge_label("yes\n") == (1, True)
    assert judge_label("no") == (0, True)
    assert judge_label("yes but") == (0, False)
    assert judge_label("") == (0, False)
