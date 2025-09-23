# Football Video Analyser

Pipeline and tooling for turning grassroots 7-a-side match footage into structured match insights (shots, saves, passes, interceptions, tackles, goals, possession segments, highlight reels).

## Getting Started

1. **Install [`uv`](https://github.com/astral-sh/uv)** (fast Python package manager):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source "$HOME/.local/bin/env"
   ```
2. **Create the project environment (Python 3.11):**
   ```bash
   uv venv --python 3.11
   source .venv/bin/activate
   ```
3. **Install the project in editable mode with dev tooling:**
   ```bash
   uv pip install -e .[dev]
   ```
4. **(Optional) Enable git hooks:**
   ```bash
   pre-commit install
   ```
5. **Run quality checks:**
   ```bash
   ruff .
   black --check .
   pytest
   ```

### Video Library Index

Generate a metadata snapshot of your footage:

```bash
uv run football-video-analyser video-index \
  --root "/mnt/e/Recordings/Old Wilsonians/U9 Forbes" \
  --root "/mnt/e/Recordings/Old Wilsonians/U10 Bradbury" \
  --output data/video-index.json
```

Append `--save-to-db` to upsert the filtered metadata into PostgreSQL alongside writing the optional JSON/CSV export.

The command logs a summary to the console and writes rich metadata (resolution, fps, frame count, bitrate, recorded timestamp) to JSON/CSV for later ingestion into PostgreSQL or other stores. Clips shorter than 40 minutes are skipped by default—adjust with `--min-duration-minutes` if you need a different threshold.

### Baseline Detection & Tracking

Install a GPU-enabled PyTorch build that matches your CUDA version (RTX 3090 + CUDA 12.1 shown below):

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Run YOLOv8 + ByteTrack directly from the CLI:

```bash
uv run football-video-analyser detect \
  /mnt/e/Recordings/Old\ Wilsonians/U9\ Forbes/20250518\ -\ Orpington\ FC.mp4 \
  --device cuda:0 \
  --weights yolov8x.pt \
  --output data/detections.csv
```

The command logs the first few detections and, when `--output` is provided, writes a CSV containing frame index, bounding box, confidence, and track IDs. Adjust `--classes` to restrict the labels or experiment with `--conf`/`--iou` thresholds.

## Database Setup (PostgreSQL)

This project manages its own PostgreSQL schema via SQLAlchemy and Alembic. Point the tools at your database and run the initial migration:

```bash
# Option 1: export variables in your shell / PowerShell
export FOOTBALL_VIDEO_DB_URL="postgresql+psycopg://user:password@host:5432/database"
# Optional: override the schema name (defaults to football_video)
export FOOTBALL_VIDEO_DB_SCHEMA="football_video"

uv run alembic upgrade head
```

Alternatively, copy `.env.example` to `.env`, fill in the values, and use a loader such as `python-dotenv` or your shell profile to export them before running Alembic.

This creates the `match_video` table (and schema if missing). Subsequent migrations will evolve the structure alongside the codebase without impacting your other Prisma-managed project.

If you prefer to run commands without activating the virtualenv, prefix each call with `uv run`, e.g. `uv run pytest`.

## Project Layout (initial)

```
football-video-analyser/
├── AGENTS.md                  # Guidance for agents & contributors
├── documentation/             # Project planning notes
├── src/football_video_analyser/
│   ├── analytics/             # Metrics, reports, highlight generation
│   ├── annotation/            # Label schemas, QC tooling
│   ├── data_ingest/           # Video ingest and normalization
│   ├── models/                # Detection, tracking, event models
│   ├── pipeline/              # Orchestration and workflows
│   ├── ui/                    # Local dashboards or review tools
│   └── utils/                 # Shared helpers
├── tests/                     # Pytest suites (CPU-friendly smoke tests)
├── main.py                    # Temporary entry point stub
├── pyproject.toml             # Project metadata & dependencies
└── README.md
```

Large assets (raw videos, model checkpoints) should live outside the repository—store paths in manifests or configure the ingest stage to read from your local media library.

## Next Steps

- Configure data locations and ingest utilities under `src/football_video_analyser/data_ingest/`.
- Stand up annotation tooling (e.g. CVAT) and document schemas in `annotation/`.
- Add continuous integration to run the lint/format/test commands automatically.
