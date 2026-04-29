# GreenDIGIT ECML PKDD 2026 Challenge

This repository contains starter kits for two related tasks:

- [Task A](task-a/README.md): forecast site-level energy and carbon signals, plus missing-signal and peak-event detection.
- [Task B](task-b/README.md): use Task A forecasts in a forecast-driven sustainable job scheduling simulator.

Shared data lives in [data/](data/). Task A and Task B should be installed in
separate virtual environments because they expose separate Python packages and
CLIs.

Related repositories:
- [GreenDIGIT-project](https://github.com/GreenDIGIT-project)

*This work is funded from the European Union’s Horizon Europe research and innovation programme through the [GreenDIGIT project](https://greendigit-project.eu/), under the grant agreement No. [101131207](https://cordis.europa.eu/project/id/101131207)*.

<div style="display:flex;align-items:center;width:100%;">
  <img src="static/EN-Funded-by-the-EU-POS-2.png" alt="EU Logo" width="250px">
  <img src="static/cropped-GD_logo.png" alt="GreenDIGIT Logo" width="110px" style="margin-right:100px">
</div>

## Task A Quick Start

The commands below run the reference baseline end to end. Participants may
replace the baseline with their own forecasting model and submission-generation
code, as long as the output CSVs keep the documented Task A schemas.

```bash
cd task-a
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,server]"

./.venv/bin/task-a train-baseline
./.venv/bin/task-a predict
./.venv/bin/task-a predict-detection
./.venv/bin/task-a predict-peaks
./.venv/bin/task-a evaluate \
  --forecasts outputs/forecast_submission.csv \
  --detections outputs/detection_submission.csv \
  --peaks outputs/peak_submission.csv
```

Read [task-a/guidelines.md](task-a/guidelines.md) before preparing a submission.
It defines the required files and the evaluator files participants must not
change.

## Task B Quick Start

Task B can consume Task A forecasts in two ways:

- Offline mode reads a forecast CSV. The default file is the bundled
  `data/forecast_baseline.csv`, which is only a local starter forecast source.
- Live mode calls a Task A-compatible `POST /forecast` service. This is the
  participant path for evaluating Task B with predictions from their own Task A
  model.

```bash
cd task-b
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,server]"

./.venv/bin/python -m dirac_sim simulate --offline
```

Participants should use Live Mode when running Task B with their own Task A
predictions. To run Task B against a live Task A model, start a forecast server
from `task-a` in one terminal:

```bash
cd task-a
source .venv/bin/activate
TASK_A_MODEL=outputs/baseline_model.json \
./.venv/bin/uvicorn task_a.api:app --host 0.0.0.0 --port 8000
```

Then run Task B without `--offline` in another terminal:

```bash
cd task-b
source .venv/bin/activate
./.venv/bin/python -m dirac_sim simulate \
  --api-url http://localhost:8000 \
  --scheduler dirac_sim.baselines.greedy_carbon.GreedyCarbonScheduler
```

Participants can replace the example Task A server with their own service, as
long as it implements the `POST /forecast` contract described in
[task-b/README.md](task-b/README.md). Task B uses those forecast responses
during scheduling; it does not automatically read `task-a/outputs/` unless a
compatible CSV is passed with `--forecast-csv` in offline mode.

Read [task-b/README.md](task-b/README.md) before preparing a scheduler
submission. It defines the scheduler interface, output files, and evaluator
files participants must not change.
