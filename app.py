from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open("models.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "Model is running successfully!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Replace these feature names with your own
        features = np.array([[
            data["feature1"],
            data["feature2"],
            data["feature3"],
            data["feature4"]
        ]])

        prediction = model.predict(features)

        return jsonify({
            "prediction": prediction.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
