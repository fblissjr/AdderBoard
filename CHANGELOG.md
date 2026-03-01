# Changelog

## 0.3.1

### Added
- CLI inference engine (infer.py) with Rich styled output and uniplot confidence charts
  - Rich Tables for long addition with per-digit columns and carry highlighting
  - Rich Panels for model info and side-by-side comparison
  - Unicode confidence bars and uniplot line charts in verbose mode
  - Help command (h/help) with styled command reference
  - Extracted run_compare() for deduplicated compare logic
- Test suite (test_infer.py) with 45 tests covering parsing, encoding, formatting, inference

## 0.3.0

### Added
- Trained 162-parameter submission (100% accuracy on 10,010 test cases)
  - Novel hybrid: fixed attention mask (0 params) + trained weights via AdamW
  - Architecture: 1L, d=3, 3h, ff=6, reversed LSB-first digits
  - Two-phase training: LR 0.01 exploration (50K steps) + LR 0.001 stabilization (5K steps)
  - Beats trained leaderboard leader (311 params) by 149 params
- Also qualified: 204p (mask-w12, 100%) and 456p (mask-large, 99.72%)
- Training infrastructure: train_adder.py with fixed-mask configs, train_continue.py
- Training report: reports/trained_submission.md

## 0.2.0

### Added
- Hand-coded 1L transformer adder (33 unique params, 100% accuracy)
- Gemini v2 submission (33 params, 100% accuracy) with full analysis

### Changed
- Updated reports with Gemini v2 analysis and carry-in-attention insight
- Updated session log with 1L model development

## 0.1.0

### Added
- Hand-coded 2L transformer adder (249 params, 100% accuracy on 10,010 test cases)
- Verification test dataset (test_dataset.json) with 10 edge + 10,000 random pairs
- Training script for curriculum-based approach (train.py)
- Comprehensive reports comparing Claude and Gemini approaches
