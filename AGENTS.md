# Repository Guidelines

## Project Structure & Module Organization
- `src/`: Core training, models, dataloaders, evaluation, and visualizers (e.g., `src/train.py`, `src/model.py`, `src/visualizers/`).
- `scripts/`: Standalone utilities for scoring and submission (`scripts/evaluation.py`, `scripts/generate_submission.py`).
- `data/`: Expected dataset layout (`train/`, `dev/`, `test1/`); this folder is ignored by git.
- `examples/`: Demo code and sample prediction artifacts.
- `checkpoints/`, `results/`: Local outputs and experiment artifacts (checkpoints are ignored by git).

## Build, Test, and Development Commands
- `python src/train.py --model cnn1d --epochs 10`: Train a model; auto-selects device unless `--device` is set.
- `python src/evaluation.py --features data/dev/features.pkl --labels data/dev/labels.pkl --checkpoint checkpoints/cnn1d_best.pt --model cnn1d`: Evaluate a checkpoint on labeled data and report EER.
- `python src/predict.py --features data/test1/features.pkl --checkpoint checkpoints/cnn1d_best.pt --model cnn1d --out prediction.pkl`: Generate `prediction.pkl` for evaluation/submission.
- `python scripts/evaluation.py prediction.pkl data/dev/labels.pkl`: Score predictions with the reference evaluation script.
- `python scripts/generate_submission.py test2/features.pkl prediction.pkl st123 First Last Nick`: Produce the leaderboard submission file.

## Coding Style & Naming Conventions
- Python code uses 4-space indentation and PEP 8-style naming (snake_case for functions/variables, PascalCase for classes).
- Keep file names in `snake_case.py` and mirror existing module boundaries (e.g., model logic stays in `src/model.py`).
- Preserve dataset schema conventions (`uttid`, `features`, `labels`) when adding loaders or utilities.

## Testing Guidelines
- There is no automated test suite in this repo; use the evaluation scripts as regression checks.
- When making changes, re-run EER on the dev set and note the results in your PR description.

## Commit & Pull Request Guidelines
- Commit messages are descriptive, imperative, and sentence-case (e.g., “Add benchmark report…” or “Refactor training script…”).
- PRs should include: a short summary, commands run (and EER results), and any new artifacts generated.
- Do not commit `data/` or model checkpoints (`*.pt`, `*.ckpt`); they are ignored by `.gitignore`.

## Environment & Dependencies
- No dependency lockfile is provided; manage your own environment.
- Core dependencies implied by the code include `torch`, `pandas`, `numpy`, and `rich`.

## Questions & Support
- Follow the ticket workflow in `How to Ask Questions: The Ticket System.md` when raising questions or issues.
