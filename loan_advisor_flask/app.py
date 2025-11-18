# app.py
import os
import uuid
from flask import Flask, render_template, request, send_from_directory, jsonify
import pickle
import numpy as np

from utils.pdf_generator import create_report
from utils.shap_explainer import explain_prediction
from utils.chatbot_engine import get_chatbot_reply

# Config
UPLOAD_FOLDER = "reports"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config['REPORTS_FOLDER'] = UPLOAD_FOLDER

# Load model and preprocessor
MODEL_PATH = "model.pkl"
PREPROCESSOR_PATH = "preprocessing.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(open("model.pkl", "rb"))

# Preprocessor may be a pipeline or None
preprocessor = None
if os.path.exists(PREPROCESSOR_PATH):
    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Collect form inputs (must match fields in index.html)
    form = request.form

    # Example expected numeric & categorical inputs - adapt to your model features order
    features = {
        "name": form.get("name", "").strip(),
        "age": float(form.get("age", 0)),
        "monthly_income": float(form.get("monthly_income", 0)),
        "employment_type": form.get("employment_type", "salaried"),
        "credit_score": float(form.get("credit_score", 0)),
        "loan_amount": float(form.get("loan_amount", 0)),
        "loan_purpose": form.get("loan_purpose", "other"),
        "existing_loans_count": float(form.get("existing_loans_count", 0)),
        "dependents": float(form.get("dependents", 0))
    }

    # Build numeric/categorical array in the EXACT order your preprocessor expects.
    # Update this ordering to match your training pipeline.
    raw_feature_order = [
        features["age"],
        features["monthly_income"],
        features["credit_score"],
        features["loan_amount"],
        features["existing_loans_count"],
        features["dependents"],
        features["employment_type"],   # categorical
        features["loan_purpose"]       # categorical
    ]

    # If preprocessor exists, apply transform; else assume numeric-only model
    X_in = np.array([raw_feature_order], dtype=object)
    if preprocessor is not None:
        X_proc = preprocessor.transform(X_in)
    else:
        # try converting to float array (drop categorical if present)
        # This fallback will likely fail if your model expects encoded categories.
        try:
            X_proc = np.array([[float(x) for x in raw_feature_order]])
        except Exception:
            return "Preprocessing pipeline not found and could not coerce input. Provide preprocessing.pkl", 500

    # Predict
    pred_label = model.predict(X_proc)[0]
    # get probability for positive class (assumes binary classifier and predict_proba available)
    try:
        prob = float(model.predict_proba(X_proc)[0][1])
    except Exception:
        prob = None

    # SHAP explanation: returns path to saved image (png)
    try:
        shap_img = explain_prediction(model, X_proc, preprocessor=preprocessor)
    except Exception as e:
        shap_img = None
        print("SHAP explain failed:", e)

    # Create PDF report
    report_filename = f"{uuid.uuid4().hex}_loan_report.pdf"
    report_path = os.path.join(app.config['REPORTS_FOLDER'], report_filename)
    create_report(
        out_path=report_path,
        applicant=features,
        prediction=pred_label,
        probability=prob,
        shap_image_path=shap_img
    )

    # Render result template
    return render_template(
        "result.html",
        applicant=features,
        prediction=pred_label,
        probability=(round(prob * 100, 2) if prob is not None else "N/A"),
        shap_image=os.path.basename(shap_img) if shap_img else None,
        report_filename=report_filename
    )


@app.route("/reports/<filename>")
def download_report(filename):
    return send_from_directory(app.config['REPORTS_FOLDER'], filename, as_attachment=True)


# Simple chatbot endpoint (AJAX)
@app.route("/chat_api", methods=["POST"])
def chat_api():
    data = request.json or {}
    message = data.get("message", "")
    reply = get_chatbot_reply(message)
    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
