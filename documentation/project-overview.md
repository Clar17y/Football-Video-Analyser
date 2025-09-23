# Football Video Analyser Approach

## Project Aim
Create an automated analysis pipeline for UK grassroots 7-a-side match footage that converts raw mp4 recordings into structured match insights (shots, saves, passes, interceptions, tackles, goals, possession phases, highlight clips).

## Language Choice
- **Primary language: Python.** The dominant deep-learning and computer-vision tooling (PyTorch, TorchVision, Ultralytics/YOLO, Detectron2, OpenCV, scikit-learn, pandas) is Python-first, which keeps experimentation, training, and deployment consistent.
- **Supplementary options:**
  - Lightweight APIs or dashboards can stay in Python (FastAPI, Streamlit) or branch into TypeScript/React if richer interactivity is required.
  - Performance-critical kernels can later be accelerated with C++/CUDA or Rust extensions, integrated via Python bindings once profiling shows a true bottleneck.

## Broad Technical Approach
1. **Data ingestion & cataloguing**
   - Centralize match videos with metadata (date, resolution, lighting, camera notes) and ingest routines that standardize frame rate and resolution.

2. **Annotation workflow**
   - Define labeling guidelines for events, possessions, and player involvement; use tools such as CVAT or Labelbox; verify quality with dual annotators on a pilot set.

3. **Core computer-vision stack**
   - Player and ball detection with fine-tuned YOLO/Detectron models.
   - Multi-object tracking (DeepSORT/ByteTrack) plus specialized ball trajectory tracking.
   - Event recognition via spatio-temporal models combining player tracks, ball context, and rule-based priors for early iterations.

4. **Analytics pipeline**
   - Persist per-frame features and detected events; compute stats (shots, saves, passes, interceptions, tackles, goals, possession shares) and derived metrics (xG proxies, pass completion).
   - Generate match summaries, player dashboards, and highlight reels tied to detected events.

5. **Infrastructure & MLOps**
   - Repository organized around data, models, pipelines, and analytics components.
   - Experiment tracking (MLflow/W&B), versioned datasets, CI for lint/tests, and Docker images for reproducible training/inference.
   - Active-learning loop: surface low-confidence or novel clips for re-annotation to improve models over time.

## Immediate Priorities
- Inventory existing footage (1080p60 and 4K60) and capture quality metadata.
- Set up Python environment, repo scaffolding, and basic ingest script to inspect new mp4 uploads.
- Draft annotation guidelines and select tooling for the first labeled batch.
