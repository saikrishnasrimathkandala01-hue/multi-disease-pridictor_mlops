# Multi-Disease Risk Screening — Standalone Prototype

A single self-contained HTML file. Open `index.html` directly in a
browser — no server, no build step, no dependency beyond two CDN font
requests (falls back to system fonts if offline).

## How it works

All six disease models from the `multi-disease-mlops` pipeline are
logistic regression. Each model's scaler (mean/scale) and coefficients
were exported once, offline, into a small JSON blob embedded in this
page. All six predictions run as plain JavaScript in the browser — same
scaling, same linear combination, same sigmoid, same tuned decision
threshold the real FastAPI backend (`api/main.py`) would use. Verified
to match the real trained pipeline's output exactly (see the project's
build notes).

To regenerate this page after retraining the pipeline, re-run
`src/train.py`, re-export each disease's scaler/coefficients/metrics, and
rebuild.

## What's on the page

- **Live 4-stage pipeline** — visually walks through input → preprocess →
  models → output as you run an assessment
- **Patient intake form** — demographics, vitals, labs, history, symptoms,
  grouped into tabs, plus two sample-patient buttons
- **Six independent result cards** — each disease scored separately, with
  its own risk band and probability
- **Model metrics section** — real test-set AUC/recall/precision per
  disease from the actual training run

## Limitations

- Static export of one trained snapshot — retrain the pipeline and this
  page will drift out of sync until rebuilt
- No logging, drift monitoring, or authentication (those live in the real
  pipeline's API/monitoring layer, not this static page)
- Six independent models, not a joint multi-label model — see the main
  project README for why
- Educational/research prototype only — not a diagnostic device
