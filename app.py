from flask import Flask, render_template, request
import numpy as np
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load your pre-trained model (ensure "model.pkl" exists in the same directory)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    # On loading the page, no prediction is passed
    return render_template("index.html", prediction_text=None)

@app.route("/process", methods=["POST"])
def process_form():
    try:
        # Collect all inputs from the form
        age = int(request.form["age"])
        gender = int(request.form["sex"])
        cp = int(request.form["cp"])
        trestbps = int(request.form["trestbps"])
        chol = int(request.form["chol"])
        fbs = int(request.form["fbs"])
        restecg = int(request.form["restecg"])
        thalach = int(request.form["thalach"])
        exang = int(request.form["exang"])
        oldpeak = float(request.form["oldpeak"])
        slope = int(request.form["slope"])
        ca = int(request.form["ca"])
        thal = int(request.form["thal"])

        # Prepare input array for the model
        input_features = np.array([[age, gender, cp, trestbps, chol, fbs, restecg,
                                       thalach, exang, oldpeak, slope, ca, thal]])

        # Predict and log input and prediction
        prediction = model.predict(input_features)
        print(f"Input: {input_features}")
        print(f"Prediction: {prediction[0]}")  # Log prediction value

        prediction_text = (
            "Heart Disease Risk: Low - You seem to be in good health!"
            if prediction[0] == 1 else
            "Heart Disease Risk: High - Please consult a healthcare professional."
        )

        return render_template("index.html", prediction_text=prediction_text)
    except Exception as e:
        return render_template("index.html", prediction_text=f"An error occurred: {e}")