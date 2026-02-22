from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load saved model, scaler, and columns
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return redirect(url_for("home"))
    try:
        # Get form values safely
        time = float(request.form.get("time", 0))
        amount = float(request.form.get("amount", 0))
        v1 = float(request.form.get("v1", 0))
        v2 = float(request.form.get("v2", 0))
        v3 = float(request.form.get("v3", 0))
        v4 = float(request.form.get("v4", 0))
        v14 = float(request.form.get("v14", 0))
        v17 = float(request.form.get("v17", 0))

        # Create full input with default 0
        input_dict = {col: 0.0 for col in columns}

        # Assign values
        input_dict["Time"] = time
        input_dict["Amount"] = amount
        input_dict["V1"] = v1
        input_dict["V2"] = v2
        input_dict["V3"] = v3
        input_dict["V4"] = v4
        input_dict["V14"] = v14
        input_dict["V17"] = v17

        # Convert to array
        input_array = np.array([list(input_dict.values())])

        # Scale input
        input_scaled = scaler.transform(input_array)

        # Predict
        prediction = model.predict(input_scaled)[0]

        if prediction == 1:
            result = "⚠️ Fraud Detected!"
        else:
            result = "✅ Transaction is Safe"

        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")


# IMPORTANT for Render deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
