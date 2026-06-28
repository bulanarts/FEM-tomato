"""
FEM Tomato — Flask Backend Server
===================================
Endpoints:
  GET  /api/mesh      → returns tomato_final.json
  POST /api/run-fem   → re-runs FEM pipeline and returns fresh results
  GET  /api/status    → health check

Run:
  cd backend
  python3 server.py
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import os
import subprocess
import sys

app = Flask(__name__)
CORS(app)   # allow React (localhost:5173) to talk to Flask (localhost:5000)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ══════════════════════════════════════════════════════
# GET /api/status
# Health check — React calls this first to confirm
# backend is alive before loading anything.
# ══════════════════════════════════════════════════════

@app.route("/api/status")
def status():
    return jsonify({
        "status": "ok",
        "message": "FEM Tomato backend is running"
    })


# ══════════════════════════════════════════════════════
# GET /api/mesh
# Returns the precomputed tomato_final.json
# React fetches this on page load to render the tomato.
# ══════════════════════════════════════════════════════

@app.route("/api/mesh")
def get_mesh():
    filepath = os.path.join(BASE_DIR, "tomato_final.json")

    if not os.path.exists(filepath):
        return jsonify({
            "error": "tomato_final.json not found. Run fem_tomato_session3.py first."
        }), 404

    with open(filepath) as f:
        data = json.load(f)

    return jsonify(data)


# ══════════════════════════════════════════════════════
# POST /api/run-fem
# Re-runs the full FEM pipeline with new parameters.
# React sends: { E, nu, load, pressure, drop_height }
# Python runs sessions 1→2→3, returns fresh results.
# ══════════════════════════════════════════════════════

@app.route("/api/run-fem", methods=["POST"])
def run_fem():
    params = request.get_json()

    # Extract parameters with sensible defaults
    E           = float(params.get("E",           80000))
    nu          = float(params.get("nu",           0.47))
    load        = float(params.get("load",         10.0))
    pressure    = float(params.get("pressure",     3000.0))
    drop_height = float(params.get("drop_height",  0.5))
    sigma_yield = float(params.get("sigma_yield",  15000.0))

    print(f"\n▶ Running FEM with params:")
    print(f"  E={E} Pa, nu={nu}, load={load} N")
    print(f"  pressure={pressure} Pa, drop_height={drop_height} m")
    print(f"  sigma_yield={sigma_yield} Pa")

    # Write a small config file that the FEM scripts will read
    config = {
        "E": E,
        "nu": nu,
        "load": load,
        "pressure": pressure,
        "drop_height": drop_height,
        "sigma_yield": sigma_yield
    }
    config_path = os.path.join(BASE_DIR, "fem_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f)

    try:
        mesh_exists    = os.path.exists(os.path.join(BASE_DIR, "tomato_mesh.json"))
        results_exists = os.path.exists(os.path.join(BASE_DIR, "tomato_results.json"))

        # Only re-mesh if mesh doesn't exist yet
        if not mesh_exists:
            print("  Running Session 1 (mesh)...")
            result = subprocess.run(
                [sys.executable, "Tomat.py"],
                cwd=BASE_DIR, capture_output=True, text=True, timeout=60
            )
            if result.returncode != 0:
                return jsonify({"error": "Session 1 failed", "detail": result.stderr}), 500
        else:
            print("  Skipping Session 1 (mesh already exists)")

        # Only re-solve if results don't exist yet
        if not results_exists:
            print("  Running Session 2 (solver)...")
            result = subprocess.run(
                [sys.executable, "Tomat2.py"],
                cwd=BASE_DIR, capture_output=True, text=True, timeout=120
            )
            if result.returncode != 0:
                return jsonify({"error": "Session 2 failed", "detail": result.stderr}), 500
        else:
            print("  Skipping Session 2 (results already exist)")

        # Always re-run Session 3 (uses new params)
        print("  Running Session 3 (impact + pressure + rupture)...")
        result = subprocess.run(
            [sys.executable, "Tomat3.py"],
            cwd=BASE_DIR, capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            return jsonify({"error": "Session 3 failed", "detail": result.stderr}), 500

        # Return fresh results
        filepath = os.path.join(BASE_DIR, "tomato_final.json")
        with open(filepath) as f:
            data = json.load(f)

        print("  ✓ FEM pipeline complete")
        return jsonify(data)

    except subprocess.TimeoutExpired:
        return jsonify({"error": "FEM pipeline timed out (>2 min)"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500



# ══════════════════════════════════════════════════════
# GET /api/config
# Returns current fem_config.json so React can
# display what parameters were last used.
# ══════════════════════════════════════════════════════

@app.route("/api/config")
def get_config():
    config_path = os.path.join(BASE_DIR, "fem_config.json")
    if not os.path.exists(config_path):
        # Return defaults
        return jsonify({
            "E": 80000,
            "nu": 0.47,
            "load": 10.0,
            "pressure": 3000.0,
            "drop_height": 0.5,
            "sigma_yield": 15000.0
        })
    with open(config_path) as f:
        return jsonify(json.load(f))


if __name__ == "__main__":
    print("=" * 50)
    print("  FEM Tomato — Backend Server")
    print("  http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)
