from flask import Flask, render_template, request
import os
import numpy as np
import joblib
import torch
import timm
import cv2
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import torch.nn as nn

# ------------------------------------------------------
# Initialize Flask app
# ------------------------------------------------------
app = Flask(__name__, template_folder="web_pp_templates")

# ------------------------------------------------------
# Define paths
# ------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

# ------------------------------------------------------
# Load ML models and preprocessors
# ------------------------------------------------------
xgb_model = xgb.XGBClassifier()
xgb_path = os.path.join(MODELS_DIR, "xgb_best_v34.json")
if os.path.exists(xgb_path):
    xgb_model.load_model(xgb_path)
else:
    raise FileNotFoundError(f"XGBoost model not found at {xgb_path}")

rf_model = joblib.load(os.path.join(MODELS_DIR, "rf.pkl"))
meta_model = joblib.load(os.path.join(MODELS_DIR, "meta_lr.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler_meta.pkl"))
le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

# ------------------------------------------------------
# CNN Feature Extractor (EfficientNet + ResNet)
# ------------------------------------------------------
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.eff = timm.create_model('efficientnet_b3', pretrained=False, num_classes=0)
        self.res = timm.create_model('resnet50', pretrained=False, num_classes=0)

        eff_path = os.path.join(MODELS_DIR, "effb3_best_highacc.pth")
        res_path = os.path.join(MODELS_DIR, "resnet50_repaired_best.pth")

        if os.path.exists(eff_path):
            self.eff.load_state_dict(torch.load(eff_path, map_location='cpu'), strict=False)
        else:
            print("⚠️ EfficientNet weights not found, using default.")

        if os.path.exists(res_path):
            self.res.load_state_dict(torch.load(res_path, map_location='cpu'), strict=False)
        else:
            print("⚠️ ResNet weights not found, using default.")

    def forward(self, x):
        x = x.float()  # ✅ Ensure correct type
        with torch.no_grad():
            e = self.eff(x)
            r = self.res(x)
        return torch.cat([e, r], dim=1)

# Initialize feature extractor
extractor = FeatureExtractor()
extractor.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
extractor = extractor.to(device)

# ------------------------------------------------------
# Image preprocessing
# ------------------------------------------------------
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"❌ Could not load image at {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (384, 384))
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))  # [C, H, W]
    img_tensor = torch.tensor(img).unsqueeze(0).float().to(device)  # ✅ FIXED dtype
    return img_tensor

# ------------------------------------------------------
# Routes
# ------------------------------------------------------
@app.route('/')
def home():
    return render_template('indexes.html')

@app.route('/predict', methods=['POST'])
def predict():
    # --------------------------------------------------
    # 1️⃣ Get uploaded image
    # --------------------------------------------------
    file = request.files['fundus_image']
    if not file:
        return "No file uploaded", 400

    img_path = os.path.join(BASE_DIR, file.filename)
    file.save(img_path)

    # --------------------------------------------------
    # 2️⃣ Get patient metadata
    # --------------------------------------------------
    age = float(request.form['age'])
    gender = 0 if request.form['gender'] == 'Male' else 1
    systolic = float(request.form['systolic'])
    diastolic = float(request.form['diastolic'])
    glucose = float(request.form['glucose'])
    pulse = systolic - diastolic
    MAP = diastolic + pulse / 3.0

    meta_features = np.array([[age, gender, systolic, diastolic, pulse, MAP, glucose]], dtype=np.float32)
    meta_scaled = scaler.transform(meta_features)

    # --------------------------------------------------
    # 3️⃣ Extract CNN features
    # --------------------------------------------------
    img_tensor = preprocess_image(img_path)
    with torch.no_grad():
        emb = extractor(img_tensor).cpu().numpy()

    # --------------------------------------------------
    # 4️⃣ ✅ Apply PCA (or dimensionality reducer)
    # --------------------------------------------------
    pca_path = os.path.join(MODELS_DIR, "emb_pca.pkl")
    if os.path.exists(pca_path):
        print("Applying PCA transformation to match training shape...")
        pca = joblib.load(pca_path)
        emb = pca.transform(emb)
    else:
        print("⚠️ Warning: PCA file not found, skipping reduction — may cause feature mismatch.")

    # --------------------------------------------------
    # 5️⃣ Combine image embedding + metadata
    # --------------------------------------------------
    X_input = np.hstack([emb, meta_scaled])

    # --------------------------------------------------
    # 6️⃣ Predict with ensemble (XGB + RF + Meta-model)
    # --------------------------------------------------
    try:
        xgb_pred = xgb_model.predict_proba(X_input)
        rf_pred = rf_model.predict_proba(X_input)
        meta_input = np.hstack([xgb_pred, rf_pred])
        final_pred = meta_model.predict(meta_input)
        label = le.inverse_transform(final_pred)[0]
    except Exception as e:
        return f"Prediction failed: {e}"

    # --------------------------------------------------
    # 7️⃣ Render output
    # --------------------------------------------------
    return render_template('results.html', label=label)


# ------------------------------------------------------
# Run Flask app
# ------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
