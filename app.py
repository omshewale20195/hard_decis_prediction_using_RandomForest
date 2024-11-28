from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model
try:
    model = pickle.load(open("model.pkl", "rb"))
except Exception as e:
    print(f"Error loading model: {e}")

@app.route("/")
def home():
    return render_template("index.html", prediction_text=None)

@app.route("/process", methods=["POST"])
def process_form():
    try:
        # Collect inputs
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

        # Log input
        print(f"Inputs: Age={age}, Gender={gender}, CP={cp}, Trestbps={trestbps}, Chol={chol}, FBS={fbs}, RestECG={restecg}, Thalach={thalach}, Exang={exang}, Oldpeak={oldpeak}, Slope={slope}, CA={ca}, Thal={thal}")

        # Prepare features
        input_features = np.array([[age, gender, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        print(f"Input Features: {input_features}")

        # Prediction
        prediction = model.predict(input_features)
        print(f"Prediction: {prediction[0]}")

        prediction_text = (
            "Heart Disease Risk: Low - You seem to be in good health!"
            if prediction[0] == 1 else
            "Heart Disease Risk: High - Please consult a healthcare professional."
        )

        return render_template("index.html", prediction_text=prediction_text)

    except Exception as e:
        import traceback
        print(f"Error: {traceback.format_exc()}")
        return render_template("index.html", prediction_text=f"Error: {str(e)}")
