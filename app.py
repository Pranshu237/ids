from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

rf_model = joblib.load("models/random_forest_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    features = np.array(data["features"]).reshape(1, -1)

    prediction = rf_model.predict(features)[0]

    result = "🚨 ATTACK DETECTED" if prediction == 1 else "✅ NORMAL TRAFFIC"

    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)