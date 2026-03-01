#!/usr/bin/env python3
"""CLI inference engine for AdderBoard.

Interactive tool for running addition problems through trained and hand-coded
transformer adder models with step-by-step visualization.

Usage:
    python infer.py                             # REPL with trained model
    python infer.py --model submission_1l.py    # use hand-coded model
    python infer.py -v                          # verbose (show logits/confidence)
    python infer.py --compare                   # run both models side by side
    python infer.py "123 + 456"                 # one-shot mode
"""

import argparse
import dataclasses
import importlib.util
import random
import re
import sys
import time

import torch
import torch.nn.functional as F
from rich.columns import Columns
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from uniplot import plot_to_string

console = Console()

# ---------------------------------------------------------------------------
# Sequence layout constants
# ---------------------------------------------------------------------------

N_DIGITS = 10
N_ANSWER = 11
ANSWER_START = 2 * N_DIGITS + 2  # 22


@dataclasses.dataclass
class LoadedModel:
    """Bundle a model with its metadata and detected type."""
    model: torch.nn.Module
    metadata: dict
    model_type: str


# ---------------------------------------------------------------------------
# Module loading (from verify.py pattern)
# ---------------------------------------------------------------------------

def load_submission(path: str):
    spec = importlib.util.spec_from_file_location("submission", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "build_model"):
        raise ValueError(f"{path}: must define build_model() -> (model, metadata)")
    return mod


def detect_model_type(model) -> str:
    vocab_size = model.embed.num_embeddings
    if vocab_size == 12:
        return "trained"
    elif vocab_size == 11:
        return "hand-coded"
    else:
        raise ValueError(f"Unknown model type: embed vocab size = {vocab_size}")


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def parse_problem(line: str):
    line = line.strip()
    if not line:
        return ("error", "Empty input")

    cmd = line.lower()
    if cmd == "q" or cmd == "quit":
        return ("quit",)
    if cmd == "h" or cmd == "help":
        return ("help",)
    if cmd == "v":
        return ("verbose",)
    if cmd == "c":
        return ("compare",)
    if cmd == "i":
        return ("info",)
    if cmd == "e":
        return ("edge",)
    if cmd == "r":
        return ("random", None)
    if cmd.startswith("r "):
        difficulty = cmd[2:].strip()
        return ("random", difficulty)

    m = re.match(r'^\s*(\d+)\s*\+\s*(\d+)\s*$', line)
    if not m:
        return ("error", "Could not parse input. Try: 123 + 456")

    a, b = int(m.group(1)), int(m.group(2))
    if a > 9_999_999_999 or b > 9_999_999_999:
        return ("error", "Numbers must be 0-9999999999 (10 digits max)")

    return ("add", a, b)


def _digits_lsb(n: int) -> list[int]:
    """Decompose n into N_DIGITS least-significant-first digits."""
    return [(n // (10 ** i)) % 10 for i in range(N_DIGITS)]


def encode_input(a: int, b: int, model_type: str) -> list[int]:
    a_digits = _digits_lsb(a)
    b_digits = _digits_lsb(b)

    if model_type == "trained":
        return a_digits + [10] + b_digits + [11]
    else:
        return a_digits + [10] + b_digits + [10]


def decode_output(seq: list[int]) -> int:
    c_digits = seq[ANSWER_START:ANSWER_START + N_ANSWER]
    return sum(d * (10 ** i) for i, d in enumerate(c_digits))


def compute_carries(a: int, b: int) -> tuple[list[int], list[int], list[int]]:
    """Return (carries, a_digits, b_digits). carries[i] = carry INTO position i."""
    a_digits = _digits_lsb(a)
    b_digits = _digits_lsb(b)

    carries = [0] * (N_DIGITS + 1)
    for i in range(N_DIGITS):
        total = a_digits[i] + b_digits[i] + carries[i]
        carries[i + 1] = 1 if total >= 10 else 0

    return carries, a_digits, b_digits


# ---------------------------------------------------------------------------
# Instrumented inference
# ---------------------------------------------------------------------------

def infer_step_by_step(model, a: int, b: int, model_type: str) -> dict:
    tokens = encode_input(a, b, model_type)
    device = next(model.parameters()).device
    steps = []

    # Pre-allocate buffer: input tokens + N_ANSWER generated tokens
    total_len = len(tokens) + N_ANSWER
    buf = torch.zeros(1, total_len, dtype=torch.long, device=device)
    buf[0, :len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)
    n = len(tokens)

    start = time.perf_counter()
    with torch.no_grad():
        for pos in range(N_ANSWER):
            logits = model(buf[:, :n])
            step_logits = logits[0, -1]

            digit_logits = step_logits[:10]
            probs = F.softmax(digit_logits.float(), dim=-1)

            token = step_logits.argmax().item()
            confidence = probs[token].item() if token < 10 else 0.0

            top3_vals, top3_idx = probs.topk(3)
            top3 = [(top3_idx[j].item(), top3_vals[j].item()) for j in range(3)]

            steps.append({
                "pos": pos,
                "token": token,
                "confidence": confidence,
                "top3": top3,
            })
            buf[0, n] = token
            n += 1

    elapsed_ms = (time.perf_counter() - start) * 1000
    seq = buf[0, :n].tolist()
    answer = decode_output(seq)

    return {
        "answer": answer,
        "steps": steps,
        "elapsed_ms": elapsed_ms,
        "seq": seq,
    }


# ---------------------------------------------------------------------------
# Display formatting (Rich renderables)
# ---------------------------------------------------------------------------

def format_long_addition(a: int, b: int, result: int):
    """Return a Rich renderable showing long-form addition with carries."""
    carries, _, _ = compute_carries(a, b)
    expected = a + b
    correct = result == expected
    has_carries = any(c == 1 for c in carries)

    sa = str(a)
    sb = str(b)
    sr = str(result)
    se = str(expected)
    width = max(len(sa), len(sb), len(sr), len(se)) + 1

    # Pad all digit strings to same width
    sa = sa.rjust(width)
    sb = sb.rjust(width)
    sr = sr.rjust(width)
    se = se.rjust(width)

    table = Table(
        show_header=False, show_edge=False, show_lines=False,
        box=None, padding=(0, 1),
    )
    # Label column + one column per digit position
    table.add_column("", style="dim", width=7, justify="right")
    for _ in range(width):
        table.add_column(width=2, justify="center")

    # Carry row (only if there are carries)
    if has_carries:
        # Build carry string for the width of the result
        # carries[i] is carry INTO position i (LSB=0)
        # We display MSB-first, so reverse
        carry_display = []
        for i in range(width - 1, -1, -1):
            if i < len(carries) and carries[i]:
                carry_display.append(Text("1", style="bold yellow"))
            else:
                carry_display.append(Text(" "))
        table.add_row("carry", *carry_display)

    # A row
    table.add_row("", *[Text(c) for c in sa])

    # +B row
    b_cells = [Text(c) for c in sb]
    table.add_row("+", *b_cells)

    # Separator
    table.add_row("", *[Text("-" * 2) for _ in range(width)])

    # Result row
    table.add_row("", *[Text(c, style="bold") for c in sr])

    # Status
    if correct:
        status = Text("CORRECT", style="bold green")
    else:
        table.add_row("expect", *[Text(c, style="dim") for c in se])
        status = Text("INCORRECT", style="bold red")

    return Group(table, Text(""), status)


def format_step_detail(steps: list[dict], a: int, b: int, model_type: str = "trained"):
    """Return a Rich Table showing per-step inference detail."""
    carries, a_digits, b_digits = compute_carries(a, b)

    table = Table(box=box.SIMPLE_HEAVY, padding=(0, 1))
    table.add_column("Pos", justify="right", style="dim")
    table.add_column("A", justify="center")
    table.add_column("B", justify="center")
    table.add_column("Carry", justify="center")
    table.add_column("Sum", justify="center")
    table.add_column("Token", justify="center", style="bold")
    table.add_column("Confidence", justify="left", min_width=22)

    for step in steps:
        pos = step["pos"]
        ad = a_digits[pos] if pos < 10 else 0
        bd = b_digits[pos] if pos < 10 else 0
        c_in = carries[pos]
        total = ad + bd + c_in

        conf = step["confidence"]
        bar_len = int(conf * 15 + 0.5)
        bar = "\u2588" * bar_len + "\u2591" * (15 - bar_len)
        conf_text = Text(f"{bar} {conf * 100:5.1f}%", style="green")

        carry_text = Text("1", style="bold yellow") if c_in else Text("0", style="dim")

        table.add_row(
            str(pos),
            str(ad),
            str(bd),
            carry_text,
            f"{ad}+{bd}+{c_in}={total}",
            str(step["token"]),
            conf_text,
        )

    elements = [table]

    if model_type == "hand-coded":
        note = Text(
            "Note: confidence reflects softmax of parabolic logits, "
            "not calibrated probability",
            style="dim italic",
        )
        elements.append(note)

    return Group(*elements)


def format_confidence_chart(all_steps: list[list[dict]], labels: list[str] | None = None):
    """Return uniplot confidence chart as a list of strings.

    Single series: all_steps=[[step, ...]]
    Multiple series: all_steps=[[steps1, ...], [steps2, ...]], labels=["A", "B"]
    """
    all_xs = [[float(s["pos"]) for s in steps] for steps in all_steps]
    all_ys = [[s["confidence"] * 100 for s in steps] for steps in all_steps]

    kwargs = dict(title="Confidence", y_unit=" %", x_unit="", height=8, width=50)

    if len(all_steps) == 1:
        return plot_to_string(all_ys[0], xs=all_xs[0], **kwargs)
    else:
        return plot_to_string(all_ys, xs=all_xs, legend_labels=labels or [], **kwargs)


def format_comparison(results: list[tuple[str, dict, dict]], a: int, b: int):
    """Return Rich Columns with side-by-side model panels."""
    expected = a + b
    panels = []
    answers = set()

    for label, metadata, result in results:
        params = metadata.get("params", "?")
        name = metadata.get("name", "unnamed")
        correct = result["answer"] == expected
        status = "CORRECT" if correct else "INCORRECT"
        answers.add(result["answer"])

        body_lines = []
        body_lines.append(f"Result: {result['answer']}")
        body_lines.append(f"Status: {status}")
        body_lines.append(f"Time:   {result['elapsed_ms']:.1f}ms")
        body = "\n".join(body_lines)

        border_style = "green" if correct else "red"
        panel = Panel(
            body,
            title=f"{name} ({params}p)",
            border_style=border_style,
            width=38,
        )
        panels.append(panel)

    if len(answers) == 1:
        agreement = Text("Both models agree.", style="green")
    else:
        agreement = Text("Models DISAGREE!", style="bold red")

    return Group(Columns(panels, equal=True), agreement)


def print_model_info(metadata: dict, model_type: str):
    """Print model info as a Rich Panel."""
    name = metadata.get("name", "unnamed")
    rows = []
    rows.append(f"Author:  {metadata.get('author', 'unknown')}")
    rows.append(f"Params:  {metadata.get('params', '?')}")
    rows.append(f"Arch:    {metadata.get('architecture', '?')}")
    rows.append(f"Type:    {model_type}")
    tricks = metadata.get("tricks", [])
    if tricks:
        rows.append(f"Tricks:  {', '.join(tricks)}")
    panel = Panel("\n".join(rows), title=name, border_style="blue")
    console.print(panel)


def print_help():
    """Print styled command reference table."""
    table = Table(title="Commands", box=box.ROUNDED, border_style="blue")
    table.add_column("Key", style="bold cyan", width=16)
    table.add_column("Action")

    table.add_row("123 + 456", "Run addition problem")
    table.add_row("r [easy|hard|carries]", "Random problem")
    table.add_row("e", "Next edge case")
    table.add_row("v", "Toggle verbose mode")
    table.add_row("c", "Toggle compare mode")
    table.add_row("i", "Show model info")
    table.add_row("h / help", "Show this help")
    table.add_row("q / quit", "Exit")

    console.print(table)


# ---------------------------------------------------------------------------
# Random problem generation
# ---------------------------------------------------------------------------

EDGE_CASES = [
    (0, 0),
    (0, 1),
    (9_999_999_999, 0),
    (9_999_999_999, 1),
    (9_999_999_999, 9_999_999_999),
    (5_000_000_000, 5_000_000_000),
    (1_111_111_111, 8_888_888_889),
    (1_234_567_890, 9_876_543_210),
    (1, 9_999_999_999),
    (5_555_555_555, 4_444_444_445),
]


def random_problem(difficulty=None, rng=None):
    if rng is None:
        rng = random.Random()

    if difficulty == "easy":
        a = rng.randint(0, 999)
        b = rng.randint(0, 999)
    elif difficulty == "hard":
        a = rng.randint(5_000_000_000, 9_999_999_999)
        b = rng.randint(5_000_000_000, 9_999_999_999)
    elif difficulty == "carries":
        a = 9_999_999_999
        b = rng.randint(1, 9_999_999_999)
    else:
        a = rng.randint(0, 9_999_999_999)
        b = rng.randint(0, 9_999_999_999)

    return a, b


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------

def run_one(lm: LoadedModel, a, b, verbose=False):
    expected = a + b
    result = infer_step_by_step(lm.model, a, b, lm.model_type)
    correct = result["answer"] == expected

    if correct:
        status = Text("CORRECT", style="bold green")
    else:
        status = Text(f"INCORRECT (expected {expected})", style="bold red")

    console.print()
    console.print(
        f"  {a} + {b} = {result['answer']}  ",
        status,
        f"  [{result['elapsed_ms']:.1f}ms]",
    )
    console.print()
    console.print(format_long_addition(a, b, result["answer"]))

    if verbose:
        console.print()
        console.print(format_step_detail(result["steps"], a, b, model_type=lm.model_type))
        chart_lines = format_confidence_chart([result["steps"]])
        console.print(Panel("\n".join(chart_lines), title="Confidence", border_style="dim"))

    console.print()
    return result


def run_compare(models: list[LoadedModel], a, b, verbose=False):
    """Run comparison across multiple loaded models."""
    expected = a + b
    results = []
    for lm in models:
        r = infer_step_by_step(lm.model, a, b, lm.model_type)
        results.append((lm, r))

    # format_comparison expects (label, metadata, result) tuples
    comp_tuples = [(lm.model_type, lm.metadata, r) for lm, r in results]
    console.print(f"\n  {a} + {b} = {expected}\n")
    console.print(format_comparison(comp_tuples, a, b))

    if verbose:
        for lm, r in results:
            name = lm.metadata.get("name", "")
            console.print()
            console.print(Panel.fit(
                format_step_detail(r["steps"], a, b, model_type=lm.model_type),
                title=name,
                border_style="blue",
            ))

        if len(results) == 2:
            chart_lines = format_confidence_chart(
                [results[0][1]["steps"], results[1][1]["steps"]],
                labels=[
                    results[0][0].metadata.get("name", "model1"),
                    results[1][0].metadata.get("name", "model2"),
                ],
            )
            console.print(Panel("\n".join(chart_lines), title="Confidence", border_style="dim"))

    console.print()


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

def _label(lm: LoadedModel) -> str:
    return f"{lm.metadata.get('name', 'unnamed')} ({lm.metadata.get('params', '?')}p)"


def repl(primary: LoadedModel, verbose=False, compare: LoadedModel | None = None):
    compare_mode = compare is not None
    rng = random.Random()

    banner_lines = [f"Model: {_label(primary)}"]
    if compare_mode:
        banner_lines.append(f"Compare: {_label(compare)}")
    banner_lines.append("Type 'h' for help")

    console.print(Panel(
        "\n".join(banner_lines),
        title="AdderBoard Inference Engine",
        border_style="cyan",
    ))
    console.print()

    edge_idx = 0

    while True:
        try:
            line = input(">> ")
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        parsed = parse_problem(line)
        cmd = parsed[0]

        if cmd == "quit":
            break
        elif cmd == "error":
            console.print(f"  [red]{parsed[1]}[/red]")
            continue
        elif cmd == "help":
            print_help()
            continue
        elif cmd == "verbose":
            verbose = not verbose
            console.print(f"  Verbose: [bold]{'ON' if verbose else 'OFF'}[/bold]")
            continue
        elif cmd == "compare":
            if compare is None:
                console.print("  [yellow]Compare mode requires --compare flag at startup.[/yellow]")
            else:
                compare_mode = not compare_mode
                console.print(f"  Compare: [bold]{'ON' if compare_mode else 'OFF'}[/bold]")
            continue
        elif cmd == "info":
            print_model_info(primary.metadata, primary.model_type)
            if compare is not None:
                console.print()
                print_model_info(compare.metadata, compare.model_type)
            continue
        elif cmd == "random":
            difficulty = parsed[1]
            a, b = random_problem(difficulty, rng)
        elif cmd == "edge":
            a, b = EDGE_CASES[edge_idx % len(EDGE_CASES)]
            edge_idx += 1
        elif cmd == "add":
            a, b = parsed[1], parsed[2]
        else:
            continue

        if compare_mode and compare is not None:
            run_compare([primary, compare], a, b, verbose=verbose)
        else:
            run_one(primary, a, b, verbose=verbose)


# ---------------------------------------------------------------------------
# Batch / stdin mode
# ---------------------------------------------------------------------------

def run_batch(lm: LoadedModel, verbose=False, stream=None):
    if stream is None:
        stream = sys.stdin
    for line in stream:
        line = line.strip()
        if not line:
            continue
        parsed = parse_problem(line)
        if parsed[0] == "add":
            run_one(lm, parsed[1], parsed[2], verbose=verbose)
        elif parsed[0] == "error":
            console.print(f"  SKIP: {line} -- {parsed[1]}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "submission_trained.py"
COMPARE_MODEL = "submission_1l.py"


def _load(path: str) -> LoadedModel:
    mod = load_submission(path)
    model, metadata = mod.build_model()
    return LoadedModel(model=model, metadata=metadata, model_type=detect_model_type(model))


def main():
    parser = argparse.ArgumentParser(
        description="Interactive inference engine for AdderBoard transformer adders"
    )
    parser.add_argument("problem", nargs="?", default=None,
                        help='One-shot problem, e.g. "123 + 456"')
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Path to submission file (default: {DEFAULT_MODEL})")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show per-step logits and confidence")
    parser.add_argument("--compare", action="store_true",
                        help="Run both trained and hand-coded models side by side")
    parser.add_argument("--compare-model", default=COMPARE_MODEL,
                        help=f"Second model for compare mode (default: {COMPARE_MODEL})")
    parser.add_argument("--batch", action="store_true",
                        help="Read problems from stdin, one per line")
    args = parser.parse_args()

    primary = _load(args.model)
    compare = _load(args.compare_model) if args.compare else None

    if args.batch:
        run_batch(primary, verbose=args.verbose)
    elif args.problem:
        parsed = parse_problem(args.problem)
        if parsed[0] == "add":
            if compare is not None:
                run_compare([primary, compare], parsed[1], parsed[2], verbose=args.verbose)
            else:
                run_one(primary, parsed[1], parsed[2], verbose=args.verbose)
        elif parsed[0] == "error":
            console.print(f"[red]Error: {parsed[1]}[/red]", file=sys.stderr)
            sys.exit(1)
    else:
        repl(primary, verbose=args.verbose, compare=compare)


if __name__ == "__main__":
    main()
