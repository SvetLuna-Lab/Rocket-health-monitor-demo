# Rocket-health-monitor-demo

Small demo of **rocket system health monitoring** with point anomaly detection on time series.

The repository shows how to:
- load a tiny telemetry CSV (pressure, temperature, vibration),
- run simple detectors (univariate **Z-score**, **EWMA**, multivariate **IsolationForest**),
- evaluate point-level **precision / recall / F1** against labeled anomalies,
- save a JSON report with predictions and metrics.

This is a minimal, dependency-light example for time-series anomaly detection.

---

## Repository structure

```text
rocket-health-monitor-demo/
├─ data/
│  └─ telemetry_small.csv
├─ src/
│  ├─ __init__.py
│  ├─ data_utils.py
│  ├─ detectors.py
│  ├─ metrics.py
│  └─ run_monitoring_experiment.py
├─ tests/
│  ├─ __init__.py
│  ├─ test_detectors.py
│  └─ test_metrics.py
├─ requirements.txt
├─ .gitignore
└─ README.md


Data

data/telemetry_small.csv contains:

t — time index

pressure — chamber/line pressure (synthetic)

temperature — structural/line temperature (synthetic)

vibration — vibration magnitude (synthetic)

is_anomaly — point label: 0 = normal, 1 = anomaly

The dataset is synthetic and only for demonstration.


Quick start

Install dependencies:

pip install -r requirements.txt


Run the experiment (prints metrics and saves a JSON report):

python src/run_monitoring_experiment.py


Console output will look like:

=== Rocket health monitoring (point anomalies) ===
zscore_pressure     P=1.000  R=1.000  F1=1.000
ewma_vibration      P=1.000  R=1.000  F1=1.000
iforest_all         P=0.667  R=1.000  F1=0.800

Best by F1: zscore_pressure (F1=1.000)



A detailed report is saved to monitoring_results.json:

per-detector metrics,

best detector by F1,

timestamps, true labels, and predictions.



Tests

Run from the project root:

python -m unittest discover -s tests


test_detectors.py — basic checks for Z-score, EWMA and IsolationForest

test_metrics.py — sanity check for precision/recall/F1 computation



Detectors (brief)

Z-score (univariate): flags points with |z| ≥ threshold (default 3.0).

EWMA (univariate): flags if |x − EWMA| exceeds k_sigma × σ(residuals).

IsolationForest (multivariate): tree-based isolation of outliers over [pressure, temperature, vibration].



Extending the demo

Add more sensors (flow, thrust, valve current) and create multivariate rules.

Introduce event-level metrics (group contiguous anomalies into events).

Tune contamination for IsolationForest on a validation split.

Add sliding-window features (rolling mean/std, deltas).

Export plots (matplotlib) to visualize flagged points.

