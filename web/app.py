import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from spellchecker.pipeline import SpellcheckerPipeline

app = Flask(__name__, template_folder="templates", static_folder="static")
pipeline = SpellcheckerPipeline()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "bertu_loaded": pipeline.neural_runtime.loaded})


@app.route("/check", methods=["POST"])
def check():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    result = pipeline.check(text)
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5004))
    app.run(host="127.0.0.1", port=port, debug=False)
