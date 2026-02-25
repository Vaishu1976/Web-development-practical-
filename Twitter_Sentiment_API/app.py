from flask import Flask, request, jsonify
import joblib
import re

app = Flask(__name__)

# Load trained model
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

@app.route("/")
def home():
    return "Twitter Sentiment Analysis API is running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text")

    if not text:
        return jsonify({"error": "Text is required"}), 400

    cleaned_text = clean_text(text)
    text_vector = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vector)[0]

    return jsonify({
        "input_text": text,
        "sentiment": prediction
    })

if __name__ == "__main__":
    app.run(debug=True)
