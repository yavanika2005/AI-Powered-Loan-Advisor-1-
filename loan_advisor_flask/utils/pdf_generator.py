# utils/pdf_generator.py
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import datetime
import os

def create_report(out_path, applicant, prediction, probability, shap_image_path=None):
    """
    Creates a PDF report summarizing applicant data, prediction and (optional) SHAP image.
    - out_path: path to write PDF
    - applicant: dict of input fields
    - prediction: model label
    - probability: float between 0-1 or None
    - shap_image_path: optional path to a PNG
    """
    c = canvas.Canvas(out_path, pagesize=A4)
    width, height = A4
    margin = 40
    y = height - margin

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "Loan Eligibility Report")
    c.setFont("Helvetica", 10)
    c.drawString(margin, y - 18, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 40

    # Applicant details
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Applicant Details")
    y -= 18
    c.setFont("Helvetica", 10)
    for k, v in applicant.items():
        c.drawString(margin, y, f"{k}: {v}")
        y -= 14
        if y < 150:
            c.showPage()
            y = height - margin

    # Prediction
    y -= 8
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Model Decision")
    y -= 18
    c.setFont("Helvetica", 10)
    c.drawString(margin, y, f"Prediction: {prediction}")
    y -= 14
    c.drawString(margin, y, f"Probability (positive): {round(probability*100,2) if probability is not None else 'N/A'}%")
    y -= 24

    # SHAP image
    if shap_image_path and os.path.exists(shap_image_path):
        try:
            # place image on the page
            img = ImageReader(shap_image_path)
            img_w = 420
            img_h = 220
            c.drawImage(img, margin, y - img_h, width=img_w, height=img_h)
            y -= img_h + 10
        except Exception as e:
            print("Failed to embed shap image:", e)

    c.showPage()
    c.save()
    return out_path
