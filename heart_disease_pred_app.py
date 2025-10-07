import joblib 
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np 

app = Flask(__name__)
CORS(app)

model = joblib.load("heart_disease_pred.pkl")  # غيّري الاسم لو مختلف

@app.route("/")
def home():
    return "Heart Disease Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)[0]
    probability = float(model.predict_proba(features)[0][1]) if hasattr(model, "predict_proba") else None
    return jsonify({'prediction': int(prediction), 'probability': probability})

if __name__ == "__main__":
    app.run(debug=True)
