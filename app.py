from flask import Flask, request, jsonify
from flask_cors import CORS
from pipeline import SISPipeline

app = Flask(__name__)
CORS(app)  # critical for cross-origin requests

pipeline = SISPipeline(n_tiles=60)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400
    result = pipeline.ask(question)
    # Convert matched_tiles to serializable format
    result["matched_tiles"] = [
        {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in t.items()}
        for t in result["matched_tiles"]
    ]
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)