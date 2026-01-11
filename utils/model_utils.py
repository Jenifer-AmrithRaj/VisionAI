"""
File: utils/model_utils.py
Purpose: Unified CNN + ML fusion predictor utilities for VisionAI
Status: Final stable version (with SHAP/LIME compatibility)
"""

import os
import torch
import torch.nn.functional as F
from torchvision import transforms, models
import numpy as np
import cv2
import timm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABELS = ["NO_DR", "MILD", "MODERATE", "SEVERE", "PDR"]

MODEL_PATHS = {
    "efficientnet": "modelss/best_efficientnetb0.pth",
    "resnet": "modelss/best_resnet50.pth",
    "vit": "modelss/best_vit_small_final.pth",
}

# ---------------------------------------------------------------------
# --- Image Preprocessing ---------------------------------------------
# ---------------------------------------------------------------------
def _clahe_transform_cv2(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def preprocess_and_save(image_path, out_path, size=(512, 512)):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = _clahe_transform_cv2(img)
    img = cv2.resize(img, size)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return out_path


def load_image_tensor(image_path, model_type="efficientnet"):
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    size_map = {"efficientnet": (512, 512), "resnet": (384, 384), "vit": (224, 224)}
    target_size = size_map.get(model_type, (224, 224))
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = _clahe_transform_cv2(img)
    img = cv2.resize(img, target_size)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0).to(DEVICE)

# ---------------------------------------------------------------------
# --- CNN Loading -----------------------------------------------------
# ---------------------------------------------------------------------
def load_cnn_models(device=DEVICE):
    models_dict = {}
    try:
        effnet = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(LABELS))
        effnet.load_state_dict(torch.load(MODEL_PATHS["efficientnet"], map_location=device), strict=False)
        effnet.to(device).eval()
        models_dict["efficientnet"] = effnet
    except Exception as e:
        print("⚠️ EfficientNet load failed:", e)

    try:
        resnet = models.resnet50(weights=None)
        resnet.fc = torch.nn.Linear(resnet.fc.in_features, len(LABELS))
        resnet.load_state_dict(torch.load(MODEL_PATHS["resnet"], map_location=device), strict=False)
        resnet.to(device).eval()
        models_dict["resnet"] = resnet
    except Exception as e:
        print("⚠️ ResNet load failed:", e)

    try:
        vit = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=len(LABELS))
        vit.load_state_dict(torch.load(MODEL_PATHS["vit"], map_location=device), strict=False)
        vit.to(device).eval()
        models_dict["vit"] = vit
    except Exception as e:
        print("⚠️ ViT load failed:", e)

    if models_dict:
        print(f"✅ CNN models loaded successfully: {list(models_dict.keys())}")
    else:
        print("⚠️ No CNN models loaded; fallback mode.")
    return models_dict

# ---------------------------------------------------------------------
# --- CNN Prediction --------------------------------------------------
# ---------------------------------------------------------------------
def predict_cnn(models_dict, image_path, device=DEVICE):
    if not models_dict:
        avg_probs = np.ones(len(LABELS)) / len(LABELS)
        return avg_probs, "UNKNOWN", float(np.max(avg_probs))

    results = []
    for name, model in models_dict.items():
        try:
            tensor = load_image_tensor(image_path, name)
            with torch.no_grad():
                logits = model(tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                results.append(probs)
        except Exception as e:
            print(f"⚠️ {name} failed:", e)

    if not results:
        avg_probs = np.ones(len(LABELS)) / len(LABELS)
    else:
        avg_probs = np.mean(results, axis=0)
        avg_probs = np.clip(avg_probs, 1e-8, 1)
        avg_probs = avg_probs / np.sum(avg_probs)

    pred_idx = int(np.argmax(avg_probs))
    return avg_probs, LABELS[pred_idx], float(np.max(avg_probs))

# ---------------------------------------------------------------------
# --- ML Metadata Predictor -------------------------------------------
# ---------------------------------------------------------------------
def predict_metadata_ml(meta, ml_models, scaler=None):
    """
    Ensures metadata vector matches trained model/scaler dimensions.
    Returns individual model predictions and average probability vector.
    """
    numeric = [
        "Age", "Systolic", "Diastolic", "Glucose_Level", "BMI", "Duration",
        "Cholesterol", "HbA1c", "Physical_Activity"
    ]
    cat = ["Smoking", "Hypertension", "Family_History", "Insulin_Use", "Medication"]

    # Build base numeric + categorical feature vector
    features = []
    for k in numeric:
        try:
            features.append(float(meta.get(k, 0)))
        except Exception:
            features.append(0.0)
    for k in cat:
        val = str(meta.get(k, "")).strip().lower()
        features.append(1.0 if val in ["yes", "true", "1"] else 0.0)

    X = np.array(features).reshape(1, -1)

    # Align to scaler or model expected feature count
    target_len = None
    if scaler is not None and hasattr(scaler, "mean_"):
        target_len = len(scaler.mean_)
    else:
        for m in ml_models.values():
            if hasattr(m, "n_features_in_"):
                target_len = m.n_features_in_
                break

    if target_len is not None and X.shape[1] != target_len:
        diff = target_len - X.shape[1]
        if diff > 0:
            X = np.pad(X, ((0, 0), (0, diff)), constant_values=0)
        elif diff < 0:
            X = X[:, :target_len]
        print(f"✅ Adjusted metadata vector to {target_len} features (auto-aligned).")

    # Apply scaler safely
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception as e:
            print("⚠️ Scaling failed:", e)

    preds = {}
    for name, model in ml_models.items():
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[0]
            else:
                raw = np.array(model.predict(X))
                probs = raw / (np.sum(raw) + 1e-8)
            probs = np.clip(probs, 1e-8, 1)
            probs = probs / np.sum(probs)
            preds[name] = probs.tolist()
        except Exception as e:
            print(f"⚠️ {name} ML model failed:", e)
            preds[name] = [0.2] * len(LABELS)

    avg_probs = np.mean(list(preds.values()), axis=0)
    avg_probs = np.clip(avg_probs, 1e-8, 1)
    avg_probs = avg_probs / np.sum(avg_probs)
    return preds, avg_probs

# ---------------------------------------------------------------------
# --- Fusion ----------------------------------------------------------
# ---------------------------------------------------------------------
def fuse_predictions(cnn_probs, ml_probs):
    """
    Weighted fusion of CNN + ML predictions.
    Normalizes and guards against NaN/Inf values.
    """
    cnn_vec = np.array(cnn_probs, dtype=float)
    ml_vec = np.array(ml_probs, dtype=float)

    if cnn_vec.shape != ml_vec.shape:
        ml_vec = np.pad(ml_vec, (0, cnn_vec.size - ml_vec.size), mode="constant")

    fused = 0.85 * cnn_vec + 0.15 * ml_vec
    fused = np.clip(fused, 1e-8, 1)
    fused = fused / np.sum(fused)

    idx = int(np.argmax(fused))
    conf = float(np.max(fused))
    risk = float(np.sum(fused[1:]) / (np.sum(fused) + 1e-8))
    return fused, LABELS[idx], conf, risk
