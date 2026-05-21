"""Tests for harness/scorer.py."""

import pytest

from harness.scorer import (
    extract_final_answer,
    score_answer,
    compute_problem_stats,
    _exact_match,
    _numeric_match,
    _parse_number,
    _clean_answer,
    _normalize_list,
    _list_match,
)


# ---------------------------------------------------------------------------
# extract_final_answer
# ---------------------------------------------------------------------------

class TestExtractFinalAnswer:
    def test_extracts_labelled_answer(self):
        text = "I analysed the data.\n\nFINAL ANSWER: T regulatory cell"
        assert extract_final_answer(text) == "T regulatory cell"

    def test_case_insensitive_label(self):
        assert extract_final_answer("final answer: BRCA2") == "BRCA2"

    def test_mixed_case_label(self):
        assert extract_final_answer("Final Answer: 42.5") == "42.5"

    def test_bold_markdown_label(self):
        # Markdown bold wrapping — label still matched, bold stripped in answer
        text = "**FINAL ANSWER: gene X**"
        result = extract_final_answer(text)
        assert "gene X" in result

    def test_fullwidth_colon(self):
        # Unicode fullwidth colon (：) accepted by the pattern
        assert extract_final_answer("FINAL ANSWER： yes") == "yes"

    def test_fallback_to_last_line_when_no_label(self):
        text = "Step 1: do X\nStep 2: do Y\nThe answer is Z"
        assert extract_final_answer(text) == "The answer is Z"

    def test_ignores_trailing_blank_lines(self):
        text = "conclusion: foo\n\n\n"
        assert extract_final_answer(text) == "conclusion: foo"

    def test_empty_string_returns_empty(self):
        assert extract_final_answer("") == ""

    def test_multiline_answer_captured_on_same_line(self):
        text = "preamble\nFINAL ANSWER: cell type A\nsome trailing text"
        result = extract_final_answer(text)
        assert result.startswith("cell type A")


# ---------------------------------------------------------------------------
# _clean_answer
# ---------------------------------------------------------------------------

class TestCleanAnswer:
    def test_strips_trailing_double_asterisk(self):
        assert _clean_answer("Homo sapiens**") == "Homo sapiens"

    def test_strips_leading_double_asterisk(self):
        assert _clean_answer("**Homo sapiens") == "Homo sapiens"

    def test_strips_bold_wrapping(self):
        assert _clean_answer("**Homo sapiens**") == "Homo sapiens"

    def test_strips_single_asterisk_italic(self):
        assert _clean_answer("*Homo sapiens*") == "Homo sapiens"

    def test_strips_underscore_bold(self):
        assert _clean_answer("__BRCA2__") == "BRCA2"

    def test_plain_text_unchanged(self):
        assert _clean_answer("Homo sapiens") == "Homo sapiens"

    def test_empty_string_unchanged(self):
        assert _clean_answer("") == ""

    def test_extract_final_answer_strips_markdown(self):
        text = "FINAL ANSWER: Plasmodium falciparum**"
        assert extract_final_answer(text) == "Plasmodium falciparum"

    # SC-2: underscore regression tests
    def test_preserves_internal_underscore(self):
        assert _clean_answer("Sample_01") == "Sample_01"

    def test_preserves_underscores_in_list(self):
        assert _clean_answer("[Sample_01, Sample_02, Sample_08]") == "[Sample_01, Sample_02, Sample_08]"

    # SC-3: bold wrapping around identifier with underscore
    def test_strips_bold_wrapping_around_identifier(self):
        assert _clean_answer("**Sample_01**") == "Sample_01"


# ---------------------------------------------------------------------------
# extract_final_answer underscore regression (SC-2)
# ---------------------------------------------------------------------------

class TestExtractFinalAnswerUnderscore:
    def test_preserves_underscore_in_identifier(self):
        assert extract_final_answer("FINAL ANSWER: Sample_01") == "Sample_01"

    def test_preserves_underscores_in_list_answer(self):
        text = "FINAL ANSWER: [Sample_01, Sample_02, Sample_08]"
        assert extract_final_answer(text) == "[Sample_01, Sample_02, Sample_08]"


# ---------------------------------------------------------------------------
# _exact_match
# ---------------------------------------------------------------------------

class TestExactMatch:
    def test_identical_strings(self):
        assert _exact_match("BRCA2", "BRCA2") is True

    def test_case_insensitive(self):
        assert _exact_match("brca2", "BRCA2") is True

    def test_strips_whitespace(self):
        assert _exact_match("  foo  ", "foo") is True

    def test_different_strings(self):
        assert _exact_match("BRCA1", "BRCA2") is False


# ---------------------------------------------------------------------------
# _parse_number
# ---------------------------------------------------------------------------

class TestParseNumber:
    def test_integer(self):
        assert _parse_number("42") == 42.0

    def test_float(self):
        assert _parse_number("3.14") == pytest.approx(3.14)

    def test_negative(self):
        assert _parse_number("-7.5") == pytest.approx(-7.5)

    def test_scientific_notation(self):
        assert _parse_number("1.2e3") == pytest.approx(1200.0)

    def test_returns_none_when_no_number(self):
        assert _parse_number("no number here") is None

    def test_returns_none_when_multiple_numbers(self):
        # Ambiguous — two numbers present, so None
        assert _parse_number("fold change 3.5 in 12 samples") is None

    def test_number_with_units(self):
        # Single number embedded in text
        assert _parse_number("fold change 3.5") == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# _numeric_match
# ---------------------------------------------------------------------------

class TestNumericMatch:
    def test_exact_numeric(self):
        assert _numeric_match("3.5", "3.5") is True

    def test_within_5_percent(self):
        assert _numeric_match("3.67", "3.5") is True   # ~4.9% error

    def test_outside_5_percent_but_within_abs(self):
        # 10.05 vs 10.0: rel error 0.5%, within tolerance
        assert _numeric_match("10.05", "10.0") is True

    def test_outside_both_tolerances(self):
        assert _numeric_match("5.0", "3.5") is False   # ~43% error, abs diff 1.5

    def test_zero_rubric_zero_predicted(self):
        assert _numeric_match("0", "0") is True

    def test_zero_rubric_nonzero_predicted(self):
        assert _numeric_match("0.5", "0") is False

    def test_returns_none_when_not_numeric(self):
        assert _numeric_match("foo", "bar") is None

    def test_returns_none_when_only_one_is_numeric(self):
        assert _numeric_match("3.5", "not a number") is None

    def test_within_abs_tolerance(self):
        # abs diff = 0.08, within 0.1
        assert _numeric_match("1.08", "1.0") is True


# ---------------------------------------------------------------------------
# _normalize_list
# ---------------------------------------------------------------------------

class TestNormalizeList:
    def test_bracket_format(self):
        result = _normalize_list("[Sample_01, Sample_02, Sample_03]")
        assert result == ["Sample_01", "Sample_02", "Sample_03"]

    def test_malformed_rubric_missing_opening_quote(self):
        # hb022 rubric: leading [' was dropped, leaving Sample_01', 'Sample_02', ...
        text = "Sample_01', 'Sample_02', 'Sample_03'"
        result = _normalize_list(text)
        assert result == ["Sample_01", "Sample_02", "Sample_03"]

    def test_single_value_returns_none(self):
        assert _normalize_list("Sample_01") is None

    def test_ellipsis_items_filtered(self):
        result = _normalize_list("[Sample_01, ..., Sample_08]")
        assert "..." not in result
        assert "Sample_01" in result

    def test_quoted_and_unquoted_produce_same_result(self):
        quoted = _normalize_list("['Sample_01', 'Sample_02']")
        unquoted = _normalize_list("[Sample_01, Sample_02]")
        assert quoted == unquoted


# ---------------------------------------------------------------------------
# _list_match
# ---------------------------------------------------------------------------

HB022_RUBRIC = (
    "Give all or nothing credit. Do not award partial credit. "
    "The answer is: Sample_01', 'Sample_02', 'Sample_03', 'Sample_04', "
    "'Sample_05', 'Sample_06', 'Sample_07', 'Sample_08' "
    "Score 1.0 if the answer meets the criteria above, 0.0 otherwise. No partial credit."
)
HB022_PREDICTED = (
    "[Sample_01, Sample_02, Sample_03, Sample_04, Sample_05, Sample_06, Sample_07, Sample_08]"
)


class TestListMatch:
    def test_hb022_scenario(self):
        assert _list_match(HB022_PREDICTED, HB022_RUBRIC) is True

    def test_both_bracket_format_match(self):
        assert _list_match("[A, B, C]", "[A, B, C]") is True

    def test_same_length_different_items_returns_false(self):
        assert _list_match("[Sample_01, Sample_02]", "[Sample_01, Sample_03]") is False

    def test_different_lengths_returns_none(self):
        assert _list_match("[Sample_01, Sample_02, Sample_03]", "[Sample_01, Sample_02]") is None

    def test_single_value_non_list_returns_none(self):
        assert _list_match("Homo sapiens", "Homo sapiens") is None


# ---------------------------------------------------------------------------
# score_answer (no API client — only exact + numeric paths exercised)
# ---------------------------------------------------------------------------

class TestScoreAnswer:
    def test_exact_match_scores_true(self):
        assert score_answer("BRCA2", "BRCA2") is True

    def test_case_insensitive_scores_true(self):
        assert score_answer("brca2", "BRCA2") is True

    def test_numeric_within_tolerance_scores_true(self):
        assert score_answer("3.67", "3.5") is True

    def test_clearly_wrong_scores_false_without_client(self):
        assert score_answer("wrong answer entirely", "BRCA2") is False

    def test_numeric_outside_tolerance_scores_false(self):
        assert score_answer("99", "3.5") is False

    def test_list_format_matches_despite_format_difference(self):
        # hb022: bracket format vs malformed Python-literal rubric — no client needed
        assert score_answer(HB022_PREDICTED, HB022_RUBRIC) is True


# ---------------------------------------------------------------------------
# compute_problem_stats
# ---------------------------------------------------------------------------

class TestComputeProblemStats:
    def test_empty_list_returns_empty_dict(self):
        assert compute_problem_stats([]) == {}

    def test_all_correct(self):
        stats = compute_problem_stats([True, True, True, True, True])
        assert stats["pass_at_1"] is True
        assert stats["pass_at_5"] is True
        assert stats["correct_count"] == 5
        assert stats["brittle"] is False

    def test_none_correct(self):
        stats = compute_problem_stats([False, False, False, False, False])
        assert stats["pass_at_1"] is False
        assert stats["pass_at_5"] is False
        assert stats["correct_count"] == 0
        assert stats["brittle"] is False

    def test_first_attempt_correct_only(self):
        stats = compute_problem_stats([True, False, False, False, False])
        assert stats["pass_at_1"] is True
        assert stats["correct_count"] == 1
        assert stats["brittle"] is True   # 1 correct out of 5 = brittle

    def test_brittle_threshold_is_two(self):
        stats = compute_problem_stats([False, True, True, False, False])
        assert stats["pass_at_1"] is False
        assert stats["brittle"] is True   # 2 correct = brittle

    def test_three_correct_not_brittle(self):
        stats = compute_problem_stats([True, True, True, False, False])
        assert stats["brittle"] is False

    def test_single_attempt_pass(self):
        stats = compute_problem_stats([True])
        assert stats["pass_at_1"] is True
        assert "pass_at_1" in stats

    def test_pass_at_n_key_reflects_list_length(self):
        stats = compute_problem_stats([False, False, True])
        assert "pass_at_3" in stats
        assert stats["pass_at_3"] is True
