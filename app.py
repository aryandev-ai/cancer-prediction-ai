import os
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

if not os.path.exists("breast.joblib"):
    import train_models

models = {
    "Breast": joblib.load("breast.joblib"),
    "Lung": joblib.load("lung.joblib"),
    "Oral": joblib.load("oral.joblib"),
    "Thyroid": joblib.load("thyroid.joblib")
}
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    cancer = request.form["type"]
    model = models[cancer]

    values = [
        float(request.form["f1"]),
        float(request.form["f2"]),
        float(request.form["f3"]),
        float(request.form["f4"]),
        float(request.form["f5"])
    ]

    import numpy as np

    needed = model.n_features_in_

    while len(values) < needed:
        values.append(0)

    sample = np.array(values).reshape(1, -1)

    prediction = model.predict(sample)[0]
    prob = model.predict_proba(sample)[0][1]

    result = "High Risk" if prediction == 1 else "Low Risk"

    confidence = round(prob * 100, 2)

    return render_template(
        "result.html",
        result=result,
        cancer=cancer,
        confidence=confidence
    )

app.run(debug=True)


