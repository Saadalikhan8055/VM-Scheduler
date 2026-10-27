from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

from flask import Flask, Response, abort, jsonify, render_template, send_from_directory, stream_with_context
from werkzeug.utils import safe_join

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency fallback
    pd = None

app = Flask(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
EXPERIMENT1_CSV = RESULTS_DIR / "experiment1_initial_placement_comparison.csv"
EXPERIMENT2_CSV = RESULTS_DIR / "experiment2_consolidation_benefit.csv"

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


@app.get("/")
def dashboard():
    return render_template("dashboard.html")


@app.get("/run/<script_name>")
def run_script(script_name: str):
    rel_path = SCRIPT_MAP.get(script_name)
    if rel_path is None:
        return Response("Invalid script name", status=400, mimetype="text/plain")

    script_path = (PROJECT_ROOT / rel_path).resolve()

    def generate():
        try:
            if not script_path.exists() or not script_path.is_file():
                yield f"data: [ERROR] Script file not found: {script_path.name}\n\n"
                yield "data: [DONE]\n\n"
                return

            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            if process.stdout is None:
                yield "data: [ERROR] Could not start script\n\n"
                yield "data: [DONE]\n\n"
                return

            try:
                for line in process.stdout:
                    yield f"data: {line.rstrip()}\n\n"
                return_code = process.wait()
                if return_code != 0:
                    yield f"data: [ERROR] Script exited with code {return_code}\n\n"
            finally:
                if process.stdout:
                    process.stdout.close()
        except Exception as exc:
            yield f"data: [ERROR] {exc}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    return Response(stream_with_context(generate()), mimetype="text/event-stream", headers=headers)


@app.get("/results-list")
def results_list():
    if not RESULTS_DIR.exists():
        return jsonify([])

    png_files = sorted(path.name for path in RESULTS_DIR.rglob("*.png") if path.is_file())
    return jsonify(png_files)


@app.get("/comparison-data")
def comparison_data():
    if not EXPERIMENT1_CSV.exists() or not EXPERIMENT2_CSV.exists():
        return jsonify({"error": "Run Full Comparison first"})

    def read_csv_rows(path: Path):
        if pd is not None:
            df = pd.read_csv(path)
            return df

        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return list(reader)

    try:
        placement_df = read_csv_rows(EXPERIMENT1_CSV)
        consolidation_df = read_csv_rows(EXPERIMENT2_CSV)
    except Exception as exc:  # pragma: no cover - runtime dependency path
        return jsonify({"error": f"Unable to read comparison data: {exc}"})

    def get_row(frame, algorithm_name: str):
        if pd is not None:
            rows = frame[frame["Algorithm"].astype(str).str.lower() == algorithm_name.lower()]
            if rows.empty:
                return None
            return rows.iloc[0]

        for row in frame:
            if str(row.get("Algorithm", "")).strip().lower() == algorithm_name.lower():
                return row
        return None

    def as_int(value):
        return int(float(value))

    def as_float(value):
        return float(value)

    our_row = None
    if pd is not None:
        our_rows = consolidation_df[consolidation_df["Algorithm"].astype(str).str.contains("our_algorithm_after_consolidation", case=False, na=False)]
        if not our_rows.empty:
            our_row = our_rows.iloc[0]
    else:
        for row in consolidation_df:
            if "our_algorithm_after_consolidation" in str(row.get("Algorithm", "")):
                our_row = row
                break

    if our_row is None:
        return jsonify({"error": "Run Full Comparison first"})

    result = {
        "our_algorithm": {
            "active": as_int(our_row["Active PMs"]),
            "idle": as_int(our_row["Idle PMs"]),
            "energy": as_float(our_row["Total Energy (watts)"]),
        }
    }

    for algorithm_name in ["round_robin", "first_fit", "best_fit"]:
        row = get_row(placement_df, algorithm_name)
        if row is None:
            continue
        result[algorithm_name] = {
            "active": as_int(row["Active PMs"]),
            "idle": as_int(row["Idle PMs"]),
            "energy": as_float(row["Total Energy (watts)"]),
        }

    return jsonify(result)


@app.get("/results/<path:filename>")
def serve_result(filename: str):
    safe_path = safe_join(str(RESULTS_DIR), filename)
    if safe_path is None:
        abort(404)

    abs_safe = Path(safe_path).resolve()
    abs_results = RESULTS_DIR.resolve()
    if abs_safe != abs_results and abs_results not in abs_safe.parents:
        abort(404)

    if not abs_safe.exists() or not abs_safe.is_file():
        abort(404)

    return send_from_directory(str(RESULTS_DIR), filename, as_attachment=False)


if __name__ == "__main__":
    print("Dashboard server ready. Open http://127.0.0.1:5000 in your browser.")
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)
