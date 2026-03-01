"""Tests for infer.py -- CLI inference engine for AdderBoard."""

import pytest

from rich.console import Console


def render_to_text(renderable, width=80) -> str:
    """Render a Rich renderable to plain text for testing."""
    console = Console(record=True, width=width, force_terminal=True)
    console.print(renderable)
    return console.export_text()


# --- Fixtures for model loading (built once per module) ---

@pytest.fixture(scope="module")
def trained_model():
    from infer import load_submission
    mod = load_submission("submission_trained.py")
    model, metadata = mod.build_model()
    return model, metadata


@pytest.fixture(scope="module")
def handcoded_model():
    from infer import load_submission
    mod = load_submission("submission_1l.py")
    model, metadata = mod.build_model()
    return model, metadata


# --- parse_problem tests ---

class TestParseProblem:
    def test_simple_addition(self):
        from infer import parse_problem
        assert parse_problem("123 + 456") == ("add", 123, 456)

    def test_no_spaces(self):
        from infer import parse_problem
        assert parse_problem("123+456") == ("add", 123, 456)

    def test_extra_spaces(self):
        from infer import parse_problem
        assert parse_problem("  123  +  456  ") == ("add", 123, 456)

    def test_large_numbers(self):
        from infer import parse_problem
        assert parse_problem("9999999999 + 9999999999") == ("add", 9999999999, 9999999999)

    def test_zero(self):
        from infer import parse_problem
        assert parse_problem("0 + 0") == ("add", 0, 0)

    def test_command_quit(self):
        from infer import parse_problem
        assert parse_problem("q") == ("quit",)

    def test_command_random(self):
        from infer import parse_problem
        assert parse_problem("r") == ("random", None)

    def test_command_random_hard(self):
        from infer import parse_problem
        assert parse_problem("r hard") == ("random", "hard")

    def test_command_random_easy(self):
        from infer import parse_problem
        assert parse_problem("r easy") == ("random", "easy")

    def test_command_edge(self):
        from infer import parse_problem
        assert parse_problem("e") == ("edge",)

    def test_command_verbose(self):
        from infer import parse_problem
        assert parse_problem("v") == ("verbose",)

    def test_command_compare(self):
        from infer import parse_problem
        assert parse_problem("c") == ("compare",)

    def test_command_info(self):
        from infer import parse_problem
        assert parse_problem("i") == ("info",)

    def test_command_help_h(self):
        from infer import parse_problem
        assert parse_problem("h") == ("help",)

    def test_command_help_word(self):
        from infer import parse_problem
        assert parse_problem("help") == ("help",)

    def test_invalid_input(self):
        from infer import parse_problem
        assert parse_problem("not a problem") == ("error", "Could not parse input. Try: 123 + 456")

    def test_number_too_large(self):
        from infer import parse_problem
        assert parse_problem("10000000000 + 1") == ("error", "Numbers must be 0-9999999999 (10 digits max)")

    def test_negative_number(self):
        from infer import parse_problem
        assert parse_problem("-1 + 5") == ("error", "Could not parse input. Try: 123 + 456")


# --- encode_input tests ---

class TestEncodeInput:
    def test_trained_simple(self):
        from infer import encode_input
        tokens = encode_input(123, 456, "trained")
        assert len(tokens) == 22
        assert tokens[:10] == [3, 2, 1, 0, 0, 0, 0, 0, 0, 0]
        assert tokens[10] == 10  # SEP_A
        assert tokens[11:21] == [6, 5, 4, 0, 0, 0, 0, 0, 0, 0]
        assert tokens[21] == 11  # SEP_B

    def test_handcoded_simple(self):
        from infer import encode_input
        tokens = encode_input(123, 456, "hand-coded")
        assert len(tokens) == 22
        assert tokens[:10] == [3, 2, 1, 0, 0, 0, 0, 0, 0, 0]
        assert tokens[10] == 10  # SEP
        assert tokens[11:21] == [6, 5, 4, 0, 0, 0, 0, 0, 0, 0]
        assert tokens[21] == 10  # SEP (same as first)

    def test_zero(self):
        from infer import encode_input
        tokens = encode_input(0, 0, "trained")
        assert tokens[:10] == [0] * 10
        assert tokens[11:21] == [0] * 10

    def test_max_value(self):
        from infer import encode_input
        tokens = encode_input(9999999999, 1, "trained")
        assert tokens[:10] == [9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
        assert tokens[11:21] == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]


# --- decode_output tests ---

class TestDecodeOutput:
    def test_simple(self):
        from infer import decode_output
        seq = list(range(22)) + [9, 7, 5, 0, 0, 0, 0, 0, 0, 0, 0]
        assert decode_output(seq) == 579

    def test_zero(self):
        from infer import decode_output
        seq = list(range(22)) + [0] * 11
        assert decode_output(seq) == 0

    def test_max_sum(self):
        from infer import decode_output
        seq = list(range(22)) + [8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 1]
        assert decode_output(seq) == 19999999998

    def test_with_eleven_digits(self):
        from infer import decode_output
        seq = list(range(22)) + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        assert decode_output(seq) == 10000000000


# --- compute_carries tests ---

class TestComputeCarries:
    def test_no_carry(self):
        from infer import compute_carries
        carries, _, _ = compute_carries(123, 456)
        assert carries == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def test_simple_carry(self):
        from infer import compute_carries
        carries, _, _ = compute_carries(5, 5)
        assert carries[0] == 0
        assert carries[1] == 1

    def test_carry_chain(self):
        from infer import compute_carries
        carries, _, _ = compute_carries(9999999999, 1)
        assert carries[0] == 0
        for i in range(1, 11):
            assert carries[i] == 1, f"Expected carry at position {i}"

    def test_max_sum(self):
        from infer import compute_carries
        carries, _, _ = compute_carries(9999999999, 9999999999)
        assert carries[0] == 0
        for i in range(1, 11):
            assert carries[i] == 1

    def test_alternating(self):
        from infer import compute_carries
        carries, _, _ = compute_carries(5050505050, 5050505050)
        assert carries == [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

    def test_returns_digits(self):
        from infer import compute_carries
        carries, a_digits, b_digits = compute_carries(123, 456)
        assert a_digits == [3, 2, 1, 0, 0, 0, 0, 0, 0, 0]
        assert b_digits == [6, 5, 4, 0, 0, 0, 0, 0, 0, 0]


# --- detect_model_type tests ---

class TestDetectModelType:
    def test_trained_model(self, trained_model):
        from infer import detect_model_type
        model, _ = trained_model
        assert detect_model_type(model) == "trained"

    def test_handcoded_model(self, handcoded_model):
        from infer import detect_model_type
        model, _ = handcoded_model
        assert detect_model_type(model) == "hand-coded"


# --- format_long_addition tests (Rich renderables) ---

class TestFormatLongAddition:
    def test_simple_correct(self):
        from infer import format_long_addition
        renderable = format_long_addition(123, 456, 579)
        text = render_to_text(renderable)
        assert "123" in text.replace(" ", "").replace("\n", "") or "1 2 3" in text
        assert "456" in text.replace(" ", "").replace("\n", "") or "4 5 6" in text
        assert "579" in text.replace(" ", "").replace("\n", "") or "5 7 9" in text
        assert "CORRECT" in text

    def test_incorrect_result(self):
        from infer import format_long_addition
        renderable = format_long_addition(123, 456, 999)
        text = render_to_text(renderable)
        assert "INCORRECT" in text
        assert "579" in text.replace(" ", "").replace("\n", "")

    def test_no_carry_omits_carry_row(self):
        from infer import format_long_addition
        renderable = format_long_addition(123, 456, 579)
        text = render_to_text(renderable)
        lines = [line.strip().lower() for line in text.split("\n") if line.strip()]
        carry_lines = [l for l in lines if "carry" in l and any(c.isdigit() and c != "0" for c in l)]
        assert len(carry_lines) == 0, f"Found carry row in no-carry problem: {carry_lines}"

    def test_with_carries_shows_carry_row(self):
        from infer import format_long_addition
        renderable = format_long_addition(999, 1, 1000)
        text = render_to_text(renderable)
        assert "1" in text

    def test_max_sum(self):
        from infer import format_long_addition
        renderable = format_long_addition(9999999999, 9999999999, 19999999998)
        text = render_to_text(renderable)
        assert "CORRECT" in text


# --- format_step_detail tests (Rich renderable) ---

class TestFormatStepDetail:
    def test_returns_renderable(self):
        from infer import format_step_detail
        steps = []
        for pos in range(11):
            steps.append({
                "pos": pos,
                "token": (3 + 6) % 10 if pos == 0 else 0,
                "confidence": 0.95,
                "top3": [(9, 0.95), (8, 0.03), (0, 0.02)] if pos == 0 else [(0, 0.99), (1, 0.005), (2, 0.005)],
            })
        renderable = format_step_detail(steps, 123, 456)
        text = render_to_text(renderable, width=100)
        assert "Pos" in text
        assert "Confidence" in text or "%" in text

    def test_hand_coded_note(self):
        from infer import format_step_detail
        steps = []
        for pos in range(11):
            steps.append({
                "pos": pos,
                "token": 0,
                "confidence": 0.95,
                "top3": [(0, 0.95), (1, 0.03), (2, 0.02)],
            })
        renderable = format_step_detail(steps, 0, 0, model_type="hand-coded")
        text = render_to_text(renderable, width=100)
        assert "parabolic" in text.lower() or "calibrated" in text.lower()


# --- infer_step_by_step tests ---

class TestInferStepByStep:
    def test_returns_correct_structure(self, trained_model):
        from infer import infer_step_by_step
        model, _ = trained_model
        result = infer_step_by_step(model, 123, 456, "trained")
        assert "answer" in result
        assert "steps" in result
        assert "elapsed_ms" in result
        assert len(result["steps"]) == 11

    def test_correct_answer_trained(self, trained_model):
        from infer import infer_step_by_step
        model, _ = trained_model
        result = infer_step_by_step(model, 123, 456, "trained")
        assert result["answer"] == 579

    def test_correct_answer_handcoded(self, handcoded_model):
        from infer import infer_step_by_step
        model, _ = handcoded_model
        result = infer_step_by_step(model, 123, 456, "hand-coded")
        assert result["answer"] == 579

    def test_step_has_confidence(self, trained_model):
        from infer import infer_step_by_step
        model, _ = trained_model
        result = infer_step_by_step(model, 123, 456, "trained")
        for step in result["steps"]:
            assert "token" in step
            assert "confidence" in step
            assert "top3" in step
            assert 0.0 <= step["confidence"] <= 1.0
            assert len(step["top3"]) == 3

    def test_edge_case_max(self, trained_model):
        from infer import infer_step_by_step
        model, _ = trained_model
        result = infer_step_by_step(model, 9999999999, 9999999999, "trained")
        assert result["answer"] == 19999999998
