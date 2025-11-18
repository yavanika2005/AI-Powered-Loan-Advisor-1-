# utils/shap_explainer.py
import os
import uuid
import numpy as np
import shap
import matplotlib.pyplot as plt

def explain_prediction(model, X_proc, preprocessor=None, output_dir="reports"):
    """
    Create a SHAP explanation image for the single instance in X_proc.
    Returns path to saved PNG.
    - model: trained model
    - X_proc: preprocessed numpy array (shape (1, n_features))
    - preprocessor: optional (unused here but kept for API compatibility)
    """
    os.makedirs(output_dir, exist_ok=True)
    img_name = f"shap_{uuid.uuid4().hex}.png"
    img_path = os.path.join(output_dir, img_name)

    # If Tree-based model, use TreeExplainer
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_proc)
        # For binary classification, shap_values might be array of 2 arrays; pick index 1
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_vals = shap_values[1]
        else:
            shap_vals = shap_values
    except Exception:
        # KernelExplainer fallback (slower)
        # Use a small background dataset (zeros or mean)
        background = np.zeros((1, X_proc.shape[1]))
        explainer = shap.KernelExplainer(model.predict_proba if hasattr(model, "predict_proba") else model.predict, background)
        shap_vals = explainer.shap_values(X_proc)

        if isinstance(shap_vals, list) and len(shap_vals) > 1:
            shap_vals = shap_vals[1]

    # Create bar plot of absolute SHAP values for the single sample
    try:
        vals = np.abs(shap_vals).flatten()
        feature_names = [f"f{i}" for i in range(vals.shape[0])]
        sorted_idx = np.argsort(vals)[::-1]
        topk = min(10, len(vals))

        plt.figure(figsize=(8, 4))
        plt.barh(
            [feature_names[i] for i in sorted_idx[:topk]][::-1],
            vals[sorted_idx][:topk][::-1]
        )
        plt.title("Top feature importances (|SHAP value|)")
        plt.tight_layout()
        plt.savefig(img_path, bbox_inches="tight")
        plt.close()
        return img_path
    except Exception as e:
        print("Failed to render SHAP plot:", e)
        return None
