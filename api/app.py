import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("🔥 Server file is running")

from flask import Flask, request, jsonify
from flask_cors import CORS
from src.preprocessing import preprocess_text



app = Flask(__name__)

# 🔥 أهم سطر (الحل)
CORS(app)

# ---------------------------------
def predict(text):
    return {
        "aspects": ["food", "service"],
        "aspect_sentiments": {
            "food": "positive",
            "service": "negative"
        }
    }

# ---------------------------------
@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.json
    text = data["text"]

    text = preprocess_text(text)

    result = predict(text)

    return jsonify(result)

# ---------------------------------
if __name__ == "__main__":
    print("🚀 Starting Flask...")
    app.run(debug=True)