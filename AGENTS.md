# Football Video Analyser Agents.md

This document steers AI agents and contributors working on the Football Video Analyser. Follow these instructions before making changes or submitting code.

## Guiding Principles
- **KISS – Keep It Simple, Straightforward**: Implement the simplest solution that satisfies requirements and defer complex abstractions until they are justified by usage.
- **DRY – Don’t Repeat Yourself**: Re-use existing helpers, configs, and pipelines; when duplication is inevitable, extract shared code into well-named utilities.
- **Own the lifecycle**: Every change should include the code, tests, docs, and configuration updates required for it to work end to end.
- **Evidence over assumptions**: Prefer instrumented measurements (metrics, benchmarks, experiment logs) when evaluating model or pipeline changes.
- **Protect player privacy**: Treat footage and derived data under GDPR-style safeguards; never hard-code personal data or export clips outside approved storage.

## Project Structure (expected)
- `documentation/`: Planning notes, project overview, design principles.
- `data_ingest/`: Video ingest, frame extraction, metadata normalization.
- `annotation/`: Labeling tools, schema definitions, quality control scripts.
- `models/`: Training code for detection, tracking, and event classifiers.
- `pipeline/`: Orchestration, batch jobs, and workflow definitions.
- `analytics/`: Metric computation, report builders, highlight generation.
- `ui/`: Dashboards or presentation layers (Streamlit/React) once introduced.
- `scripts/`: CLI utilities for common maintenance or evaluation tasks.
- `tests/`: Pytest suites covering critical functionality.
- `notebooks/` (optional): Exploratory analyses; keep results version-controlled if shared.

Keep heavy assets (raw videos, large model checkpoints) outside git; reference them via manifests or DVC-equivalent tracking.

## Coding Conventions
- Primary language: Python ≥3.11.
- Use type annotations and dataclasses/Pydantic models where structure matters.
- Follow `black` formatting (default profile) and `ruff` linting; resolve lint warnings unless explicitly waived.
- Prefer explicit logging via the standard `logging` module; avoid print debugging in committed code.
- Keep functions focused; max ~100 lines; extract reusable logic into modules under `utils/` or domain-specific packages.
- For deep learning code, wrap model configs with Hydra/OmegaConf-compatible structures; maintain reproducible seeds and device handling.
- Document non-obvious logic with concise comments or docstrings, especially around CV heuristics and data transforms.

## Data & Annotation Handling
- Assume videos arrive as `.mp4` at 1080p60 or 4K60; ingest code must verify fps/resolution and log anomalies.
- When the XBotGo auto-tracker loses the ball, add fallback interpolation or manual override hooks—capture these behaviors in docs/tests.
- Store annotation schemas (event labels, player metadata) centrally and version them; update guidelines alongside schema changes.

## Testing Protocols
Run these commands before requesting review or merging:

```bash
# Unit/integration tests
pytest

# Linting
ruff .

# Formatting check
black --check .

# (Optional) Type checking once configured
pyright
```

Add or update tests with every feature or bug fix. For GPU-bound components, provide CPU-friendly smoke tests or mock layers so CI stays fast.

## Pull Request Guidelines
1. Summarize the problem, approach, and validation performed (tests, benchmarks, sample outputs).
2. Reference related issues/tasks and note any data migrations or retraining required.
3. Attach key metrics (mAP, IDF1, F1 etc.) or screenshots for analytics/UI changes.
4. Keep PRs focused; unrelated refactors belong in separate submissions.
5. Ensure documentation updates land alongside behavioral changes (e.g., new event definitions, pipeline switches).

## Operational Checklist
- Ensure all automated checks above pass locally; CI failures must be resolved before merge.
- Confirm new dependencies are added to the project manifest/lockfile.
- For model updates, register experiment metadata (config, dataset snapshot, performance) in the chosen tracking system.
- Validate that added scripts handle missing GPU gracefully or provide clear error messages.

Following these guidelines keeps the football analysis pipeline maintainable, auditable, and trustworthy for coaches and players alike.
