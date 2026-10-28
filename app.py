"""Local Flask dashboard for the ML-based VM placement project.

Run from the project root:

    python app.py

then open http://127.0.0.1:5000

What this server does
---------------------
* Runs one of a fixed list of project scripts and streams its output to the
  browser line by line (Server-Sent Events).
* Runs only ONE script at a time, because the scripts read and write the same
  files in data/ models/ results/.
* Stops the running script if the browser tab is closed or refreshed.
* Hides harmless TensorFlow/protobuf warnings and shortens the very long
  "Idle PM IDs" list so the terminal panel stays readable. Add ?raw=1 to the
  /run/<script> URL to see the unfiltered output.
* Serves the CSV files in results/ as JSON so the dashboard can show tables,
  headline numbers and PM pool comparisons.
"""
from __future__ import annotations

import csv
import math
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    render_template,
    request,
    send_from_directory,
    stream_with_context,
)
from werkzeug.utils import safe_join

app = Flask(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"

# Only these scripts can be run from the browser.
SCRIPT_MAP = {
    "clustering": "src/clustering.py",
    "autoencoder": "src/autoencoder.py",
    "svm": "src/svm_classifier.py",
    "preliminary_scheduling": "src/preliminary_scheduling.py",
    "consolidation": "src/consolidation.py",
    "full_comparison": "src/experiments/run_full_comparison.py",
    "multiseed": "src/experiments/run_multiseed_experiment.py",
    "sensitivity": "src/experiments/run_sensitivity_analysis.py",
    "hyperparameter": "src/experiments/run_hyperparameter_comparison.py",
}

# CSV files the dashboard can show as tables: name -> (title, file name).
TABLES = {
    "placement": ("Experiment 1: placement only", "experiment1_initial_placement_comparison.csv"),
    "consolidation": ("Experiment 2: consolidation effect", "experiment2_consolidation_benefit.csv"),
    "multiseed": ("Multi-seed stability", "multiseed_variance_results.csv"),
    "sensitivity": ("Threshold sensitivity", "sensitivity_analysis_results.csv"),
    "hyperparameters": ("Hyperparameter checks", "hyperparameter_comparison_summary.csv"),
}

# Files from an earlier, unfair comparison (our algorithm after churn and
# consolidation vs baselines on the full workload). They are hidden from the
# gallery. Delete them from results/ as well.
HIDDEN_RESULTS = {
    "active_pms_comparison_chart.png",
    "energy_comparison_chart.png",
}

# Lines containing any of these are harmless library noise.
NOISE_MARKERS = (
    "MessageFactory",
    "oneDNN custom operations",
    "cpu_feature_guard",
    "To enable the following instructions",
    "WARNING:tensorflow:From",
    "tf.losses.sparse_softmax_cross_entropy",
    "tf.executing_eagerly_outside_functions",
    "tf.ragged.RaggedTensorValue",
)
IDLE_LIST_RE = re.compile(r"^\s*Idle PM IDs:\s*\[(.*)\]\s*$")

# One script at a time.
RUN_LOCK = threading.Lock()


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def sse(text: str) -> str:
    """Format one Server-Sent Event message."""
    return f"data: {text.replace(chr(13), '')}\n\n"


def clean_line(line: str):
    """Return the line to show, or None to hide it."""
    text = line.rstrip()
    stripped = text.strip()
    if not stripped:
        return None
    if any(marker in stripped for marker in NOISE_MARKERS):
        return None
    match = IDLE_LIST_RE.match(stripped)
    if match:
        count = len([part for part in match.group(1).split(",") if part.strip()])
        return f"Idle PM IDs: [{count} listed, names hidden in the dashboard]"
    return text


def to_float(value):
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def read_csv_table(path: Path):
    """Return (columns, rows) with every cell as a string."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.reader(handle) if row]
    if not rows:
        return [], []
    return rows[0], rows[1:]


def column_index(columns, name: str):
    wanted = name.strip().lower()
    for index, column in enumerate(columns):
        if column.strip().lower() == wanted:
            return index
    return None


def pm_metrics(columns, row):
    """Pull the standard PM metrics out of one CSV row."""

    def get(name):
        index = column_index(columns, name)
        if index is None or index >= len(row):
            return None
        return to_float(row[index])

    active, idle = get("Active PMs"), get("Idle PMs")
    return {
        "active": int(round(active)) if active is not None else None,
        "idle": int(round(idle)) if idle is not None else None,
        "energy": get("Total Energy (watts)"),
        "cpu": get("Avg CPU Utilization"),
        "memory": get("Avg Memory Utilization"),
    }


def algorithm_rows(columns, rows):
    """Yield (algorithm_name_lowercase, row) for each data row."""
    name_index = column_index(columns, "Algorithm")
    if name_index is None:
        return
    for row in rows:
        if name_index < len(row):
            yield row[name_index].strip().lower(), row


def build_placement(path: Path):
    columns, rows = read_csv_table(path)
    found = {}
    for name, row in algorithm_rows(columns, rows):
        if "round_robin" in name:
            key = "round_robin"
        elif "first_fit" in name:
            key = "first_fit"
        elif "best_fit" in name:
            key = "best_fit"
        elif "our_algorithm" in name:
            key = "our_algorithm"
        else:
            continue
        found[key] = pm_metrics(columns, row)
    return found


def build_consolidation(path: Path):
    columns, rows = read_csv_table(path)
    before = after = None
    for name, row in algorithm_rows(columns, rows):
        if "before_consolidation" in name:
            before = pm_metrics(columns, row)
        elif "after_consolidation" in name:
            after = pm_metrics(columns, row)
    if before is None or after is None:
        return None
    reduction = None
    if before["energy"] and after["energy"] is not None:
        reduction = (before["energy"] - after["energy"]) / before["energy"] * 100.0
    return {"before": before, "after": after, "reduction_pct": reduction}


def seed_stats(path: Path):
    """Mean and population std of every 'reduction' column across numeric-seed rows."""
    columns, rows = read_csv_table(path)
    seed_rows = [row for row in rows if row and row[0].strip().lstrip("-").isdigit()]
    reduction_columns = [i for i, name in enumerate(columns) if "reduction" in name.lower()]
    only_one = len(reduction_columns) == 1
    stats = []
    for index in reduction_columns:
        values = [to_float(row[index]) for row in seed_rows if index < len(row)]
        values = [v for v in values if v is not None]
        if not values:
            continue
        mean = sum(values) / len(values)
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        name = columns[index].strip()
        lower = name.lower()
        if "consolidation" in lower:
            label = "Consolidation-only reduction"
        elif "total" in lower:
            label = "Total reduction vs. initial placement"
        elif only_one and lower == "energy reduction %":
            # Older script version: it compared the state before churn with the
            # state after consolidation, so it is the total reduction.
            label = "Total reduction vs. initial placement"
        else:
            label = name
        stats.append({"label": label, "mean": mean, "std": std, "n": len(values)})
    return stats


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/")
def dashboard():
    return render_template("dashboard.html")


@app.get("/run/<script_name>")
def run_script(script_name: str):
    rel_path = SCRIPT_MAP.get(script_name)
    if rel_path is None:
        return Response("Invalid script name", status=400, mimetype="text/plain")

    script_path = (PROJECT_ROOT / rel_path).resolve()
    show_raw = request.args.get("raw") == "1"

    def generate():
        acquired = RUN_LOCK.acquire(blocking=False)
        process = None
        try:
            if not acquired:
                yield sse("[ERROR] Another script is still running. Wait for it to finish.")
            elif not script_path.is_file():
                yield sse(f"[ERROR] Script file not found: {rel_path}")
            else:
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"  # send output as it is produced
                env["PYTHONIOENCODING"] = "utf-8"
                env["TF_CPP_MIN_LOG_LEVEL"] = "2"  # quieter TensorFlow logs
                process = subprocess.Popen(
                    [sys.executable, "-u", str(script_path)],
                    cwd=str(PROJECT_ROOT),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                for raw_line in process.stdout:
                    line = raw_line.rstrip() if show_raw else clean_line(raw_line)
                    if line is not None and line != "":
                        yield sse(line)
                return_code = process.wait()
                if return_code != 0:
                    yield sse(f"[ERROR] Script exited with code {return_code}")
        except Exception as exc:  # report instead of crashing the stream
            yield sse(f"[ERROR] {exc}")
        finally:
            # Runs on normal end, errors, and when the browser disconnects.
            if process is not None:
                if process.stdout:
                    process.stdout.close()
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
            if acquired:
                RUN_LOCK.release()
        yield sse("[DONE]")

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return Response(stream_with_context(generate()), mimetype="text/event-stream", headers=headers)


@app.get("/results-list")
def results_list():
    if not RESULTS_DIR.exists():
        return jsonify([])
    names = sorted(
        path.name
        for path in RESULTS_DIR.glob("*.png")
        if path.is_file() and path.name not in HIDDEN_RESULTS
    )
    return jsonify(names)


@app.get("/comparison-data")
def comparison_data():
    """Two fair views, matching the two experiments.

    placement      Experiment 1: all four algorithms host the same VMs.
    consolidation  Experiment 2: our algorithm before vs. after consolidation,
                   with the same VMs in both states.
    """
    exp1 = RESULTS_DIR / TABLES["placement"][1]
    exp2 = RESULTS_DIR / TABLES["consolidation"][1]
    if not exp1.exists() or not exp2.exists():
        return jsonify({"error": "Run Full Comparison first"})
    try:
        placement = build_placement(exp1)
        consolidation = build_consolidation(exp2)
    except Exception as exc:
        return jsonify({"error": f"Unable to read comparison data: {exc}"})
    if not placement or consolidation is None:
        return jsonify({"error": "Run Full Comparison first"})
    return jsonify({"placement": placement, "consolidation": consolidation})


@app.get("/summary")
def summary():
    """Headline numbers for the cards at the top of the dashboard."""
    payload = {"consolidation": None, "seed_stats": []}
    exp2 = RESULTS_DIR / TABLES["consolidation"][1]
    multiseed = RESULTS_DIR / TABLES["multiseed"][1]
    try:
        if exp2.exists():
            payload["consolidation"] = build_consolidation(exp2)
    except Exception:
        payload["consolidation"] = None
    try:
        if multiseed.exists():
            payload["seed_stats"] = seed_stats(multiseed)
    except Exception:
        payload["seed_stats"] = []
    return jsonify(payload)


@app.get("/table/<name>")
def table(name: str):
    entry = TABLES.get(name)
    if entry is None:
        abort(404)
    title, filename = entry
    path = RESULTS_DIR / filename
    if not path.exists():
        return jsonify({"error": f"{filename} not found. Run the matching script first."})
    try:
        columns, rows = read_csv_table(path)
    except Exception as exc:
        return jsonify({"error": f"Unable to read {filename}: {exc}"})
    return jsonify({"title": title, "columns": columns, "rows": rows[:500]})


@app.get("/results/<path:filename>")
def serve_result(filename: str):
    safe_path = safe_join(str(RESULTS_DIR), filename)
    if safe_path is None:
        abort(404)
    resolved = Path(safe_path).resolve()
    results_root = RESULTS_DIR.resolve()
    if results_root not in resolved.parents:
        abort(404)
    if not resolved.is_file():
        abort(404)
    return send_from_directory(str(RESULTS_DIR), filename, as_attachment=False)


if __name__ == "__main__":
    print("Dashboard server ready. Open http://127.0.0.1:5000 in your browser.")
    print("Press Ctrl+C to stop. Restart the server after editing any file.")
    debug = os.environ.get("DASHBOARD_DEBUG") == "1"
    app.run(host="127.0.0.1", port=5000, debug=debug, use_reloader=False, threaded=True)
