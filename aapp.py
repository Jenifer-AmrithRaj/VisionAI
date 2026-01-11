# ============================================================
# VisionAI — Smart Retinal Disease Screening 
# ============================================================

import os
import io
import base64
import sqlite3
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for, flash
from PIL import Image, ImageFilter
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import joblib
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from flask import session


# -------------------------
# Configuration
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
GRADCAM_FOLDER = BASE_DIR / "static" / "gradcam"
REPORT_FOLDER = BASE_DIR / "static" / "reports"
MODEL_FOLDER = BASE_DIR / "models"
DB_PATH = BASE_DIR / "patient_logs.db"

for folder in (UPLOAD_FOLDER, GRADCAM_FOLDER, REPORT_FOLDER):
    folder.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 320
NUM_CLASSES = 5
DEFAULT_LABELS = ["No_DR", "Mild", "Moderate", "Severe", "PDR"]

# short mapping for presentation
MERGED_LABELS = {
    "No_DR": "No_DR",
    "Mild": "Early_DR",
    "Moderate": "Early_DR",
    "Severe": "Advanced_DR",
    "PDR": "Advanced_DR"
}

# thresholds you can tune quickly
# thresholds (tune if needed)
THRESH_TOP_CONF = 0.45        # if a single class has >=45% prob, trust it
THRESH_UNCERTAIN = 0.20       # lower => less "AI Uncertain"
# early/advanced combined-prob thresholds:
THRESH_EARLY_COMBINED = 0.45
THRESH_ADVANCED_COMBINED = 0.45


# model filenames - change to your actual filenames if different
EFF_MODEL_FILE = MODEL_FOLDER / "effb3_best_highacc.pth"
META_MODEL_FILE = MODEL_FOLDER / "xgb_best_v34.pkl"  # optional

# -------------------------
# Flask Setup
# -------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "visionai-secret-key"
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)

# -------------------------
# DB init
# -------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT, address TEXT, contact TEXT,
        age INTEGER, gender TEXT, systolic REAL, diastolic REAL,
        glucose REAL, result TEXT, confidence REAL, date TEXT, image_path TEXT
    )''')
    conn.commit(); conn.close()

init_db()

# -------------------------
# Load model(s) & pickles
# -------------------------
def load_torch_model(path: Path, model_type='eff'):
    if not path.exists():
        print("⚠️ model file missing:", path)
        return None
    try:
        if model_type == 'eff':
            model = models.efficientnet_b3(weights=None)
            in_f = int(getattr(model.classifier[1], "in_features", 0))
            model.classifier[1] = torch.nn.Linear(in_f, NUM_CLASSES)
        else:
            model = models.resnet50(weights=None)
            in_f = int(getattr(model.fc, "in_features", 0))
            model.fc = torch.nn.Linear(in_f, NUM_CLASSES)
    except Exception as e:
        print("Architecture build failed:", e); return None

    try:
        state = torch.load(str(path), map_location=DEVICE)
        # handle common wrapper keys
        if isinstance(state, dict):
            for key in ("state_dict", "model_state_dict", "model_state", "model"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]; break
        # strip 'module.' if present
        new_state = {}
        if isinstance(state, dict):
            for k, v in state.items():
                new_state[k.replace("module.", "")] = v
            state = new_state
        model.load_state_dict(state, strict=False)
        model.to(DEVICE).eval()
        print("✅ Loaded torch model:", path.name)
        return model
    except Exception as e:
        print("Failed to load state:", e); return None

def load_pickles():
    pickles = {}
    p = META_MODEL_FILE
    if p.exists():
        try:
            pickles['meta'] = joblib.load(p)
            print("✅ Loaded meta model:", p.name)
        except Exception as e:
            print("⚠️ Failed loading meta model:", e)
    # label encoder / scaler not required but try load generically
    for fname in ("label_encoder.pkl","scaler_meta.pkl"):
        f = MODEL_FOLDER / fname
        if f.exists():
            try:
                pickles[fname.split(".")[0]] = joblib.load(f)
                print("✅ Loaded:", fname)
            except:
                pass
    return pickles

IMG_MODEL = load_torch_model(EFF_MODEL_FILE, 'eff')
PICKLES = load_pickles()

# -------------------------
# Transforms & enhancement
# -------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def open_image_as_tensor(path: str):
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0).to(DEVICE)

def gentle_enhance(path: str):
    """Gentle CLAHE + slight denoise. returns new path (or same on failure)."""
    try:
        img = cv2.imread(path)
        if img is None: return path
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l,a,b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8,8))  # gentle
        cl = clahe.apply(l)
        merged = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
        out = str(Path(path).with_name(Path(path).stem + "_enhanced" + Path(path).suffix))
        # slight bilateral filter (preserve edges)
        enhanced_bgr = cv2.bilateralFilter(enhanced_bgr, d=5, sigmaColor=75, sigmaSpace=75)
        cv2.imwrite(out, enhanced_bgr)
        return out
    except Exception as e:
        print("Enhance failed:", e)
        return path

# -------------------------
# Grad-CAM (SmoothGrad-averaged + circular mask)
# -------------------------
def find_last_conv(model):
    from torch.nn import Conv2d
    last = None
    for m in model.modules():
        if isinstance(m, Conv2d):
            last = m
    return last

def fundus_circular_mask(img_arr):
    # create circular mask based on image center and radius
    h, w = img_arr.shape[:2]
    cx, cy = w//2, h//2
    r = int(min(cx, cy)*0.95)
    Y, X = np.ogrid[:h, :w]
    dist = (X - cx)**2 + (Y - cy)**2
    mask = dist <= (r*r)
    return mask.astype(np.uint8)

def smooth_gradcam(model, img_path, class_idx, n_samples=8, stdev_spread=0.15):
    """
    Stable SmoothGrad-style Grad-CAM:
      - averages n_samples Grad-CAMs on noisy inputs,
      - RESIZES each per-sample CAM to the original image size BEFORE accumulation
      - applies circular fundus mask and strong lesion thresholding,
      - writes overlay to static/gradcam and returns web path or None.
    """
    try:
        if model is None or not os.path.exists(img_path):
            print("[Grad-CAM] missing model or image.")
            return None

        # load image & sizes
        pil = Image.open(img_path).convert("RGB")
        orig_rgb = np.array(pil)  # shape (H,W,3) uint8
        h, w = orig_rgb.shape[:2]

        # find last conv layer
        target_layer = None
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                target_layer = m
        if target_layer is None:
            print("[Grad-CAM] no Conv2d found.")
            return None

        # hooks
        activations = []
        gradients = []

        def forward_hook(module, inp, out):
            activations.append(out.detach())

        def backward_hook(module, grad_in, grad_out):
            g = grad_out[0] if isinstance(grad_out, (tuple, list)) else grad_out
            gradients.append(g.detach())

        fh = target_layer.register_forward_hook(forward_hook)
        # prefer full backward hook if available
        if hasattr(target_layer, "register_full_backward_hook"):
            bh = target_layer.register_full_backward_hook(lambda m, gi, go: backward_hook(m, gi, go))
        else:
            bh = target_layer.register_backward_hook(backward_hook)

        # prepare normalization constants
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # base image resized for model input
        base_pil = pil.resize((IMG_SIZE, IMG_SIZE))
        base_arr = np.asarray(base_pil).astype(np.float32) / 255.0

        # accumulator on ORIGINAL IMAGE SIZE (to avoid resizing at the end)
        cam_accum = np.zeros((h, w), dtype=np.float32)
        valid_count = 0

        model.zero_grad()
        for i in range(max(1, int(n_samples))):
            # noise in image space
            noise = np.random.normal(0, stdev_spread, base_arr.shape).astype(np.float32)
            noisy = np.clip(base_arr + noise, 0.0, 1.0)

            # to tensor and normalize correctly (float32)
            tensor = torch.from_numpy(noisy).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            tensor = (tensor - torch.tensor(mean, device=DEVICE).view(1,3,1,1)) / torch.tensor(std, device=DEVICE).view(1,3,1,1)

            # clear hooks buffer and run
            activations.clear(); gradients.clear()
            out = model(tensor)
            if isinstance(out, (tuple, list)):
                out = out[0]
            # pick score for class_idx
            score = out[0, int(class_idx)]
            model.zero_grad()
            # backward (no retain_graph needed)
            score.backward(retain_graph=False)

            if len(activations) == 0 or len(gradients) == 0:
                continue

            act = activations[-1]   # shape (N, C, Hf, Wf)
            grad = gradients[-1]    # same channels

            # compute weights and cam (per-sample)
            weights = grad.mean(dim=(2,3), keepdim=True)           # (N, C, 1,1)
            cam = (weights * act).sum(dim=1).squeeze(0)            # (Hf, Wf)
            cam = torch.relu(cam)
            cam_np = cam.cpu().numpy().astype(np.float32)

            # normalize per-sample cam safely
            if cam_np.max() > 0:
                cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-9)
            else:
                cam_np = np.zeros_like(cam_np, dtype=np.float32)

            # RESIZE this per-sample cam to ORIGINAL image size BEFORE adding
            cam_resized = cv2.resize(cam_np, (w, h), interpolation=cv2.INTER_LINEAR)
            cam_accum += cam_resized
            valid_count += 1

        # cleanup hooks
        try:
            fh.remove(); bh.remove()
        except Exception:
            pass

        if valid_count == 0:
            print("[Grad-CAM] no valid cams computed.")
            return None

        # average
        cam_final = cam_accum / float(valid_count)
        # mask to fundus circle
        cx, cy = w // 2, h // 2
        rr = int(min(cx, cy) * 0.95)
        Y, X = np.ogrid[:h, :w]
        fundus_mask = ((X - cx)**2 + (Y - cy)**2) <= (rr * rr)
        cam_final = cam_final * fundus_mask.astype(np.float32)

        # normalize final map
        if cam_final.max() > 0:
            cam_final = (cam_final - cam_final.min()) / (cam_final.max() - cam_final.min() + 1e-9)
        else:
            cam_final = np.zeros_like(cam_final, dtype=np.float32)

        # produce lesion mask (top percent)
        perc = 85
        lesion_thresh = np.percentile(cam_final[fundus_mask], perc) if np.any(fundus_mask) else cam_final.max()
        lesion_mask = (cam_final >= lesion_thresh).astype(np.uint8)

        # morphological clean-up (use a kernel that is relative to image size)
        ksize = max(3, int(min(h, w) * 0.01))  # ~1% of smaller dim
        k = np.ones((ksize, ksize), dtype=np.uint8)
        lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, k)
        lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_CLOSE, k)

        # optional small blur for smooth edges
        lesion_mask_blur = cv2.GaussianBlur(lesion_mask.astype(np.float32), (ksize|1, ksize|1), 0)
        lesion_mask_blur = (lesion_mask_blur - lesion_mask_blur.min()) / (lesion_mask_blur.max() + 1e-9)

        # Normalize Grad-CAM (use the averaged cam_final)
        cam_smooth = (cam_final - cam_final.min()) / (cam_final.max() - cam_final.min() + 1e-8)

        # Apply color map
        heat_uint8 = np.uint8(255 * cam_smooth)
        heatmap = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Fade away low activation (remove strong blue regions)
        fade_strength = np.clip(cam_smooth, 0.4, 0.9)
        heatmap = heatmap * fade_strength[:, :, None] + (orig_rgb.astype(np.float32) / 255.0) * (1 - fade_strength[:, :, None])

        # Mask outside retina region
        gray = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY)
        _, fundus_mask = cv2.threshold(gray, 10, 1, cv2.THRESH_BINARY)
        fundus_edge = cv2.GaussianBlur(fundus_mask, (11, 11), 0)
        overlay = (heatmap * fundus_edge[:, :, None] + orig_rgb / 255.0 * (1 - fundus_edge[:, :, None]) * 0.6)

        # Boost color slightly for visibility
        overlay = np.clip(overlay * 1.2, 0, 1)

        # Convert back to uint8
        overlay = np.uint8(255 * overlay)



        # save result
        out_name = f"gradcam_{Path(img_path).stem}_{datetime.now().strftime('%H%M%S')}.png"
        out_path = GRADCAM_FOLDER / out_name
        cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        return "/static/gradcam/" + out_name

    except Exception as e:
        print("[Grad-CAM] failed:", e)
        return None


# -------------------------
# Prediction logic (robust)
# -------------------------
def safe_float(x, default=0.0):
    try: return float(x)
    except: return default
def predict(image_path: str, metadata: dict):
    """
    Returns: (label:str, confidence_percent:float [0..100], gradcam_url or None)
    """
    # 1) gentle enhancement (cheap) - keeps original too
    enhanced = gentle_enhance(image_path)

    # 2) image forward
    if IMG_MODEL is None:
        print("No image model loaded")
        return "Unknown", 0.0, None

    try:
        tensor = open_image_as_tensor(enhanced)
        with torch.no_grad():
            out = IMG_MODEL(tensor)
        if isinstance(out, (tuple, list)):
            out = out[0]
        logits = out[0].detach().cpu().numpy()
    except Exception as e:
        print("Model forward failed:", e)
        return "Unknown", 0.0, None

    # softmax with temperature (tune if needed)
    T = 1.0
    exps = np.exp(logits / T - np.max(logits / T))
    probs = exps / (exps.sum() + 1e-9)
    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx])
    top_label_raw = DEFAULT_LABELS[top_idx]
    merged_label = MERGED_LABELS.get(top_label_raw, top_label_raw)

    # 3) metadata risk
    glucose = safe_float(metadata.get("glucose", 0))
    systolic = safe_float(metadata.get("systolic", 0))
    diastolic = safe_float(metadata.get("diastolic", 0))
    age = safe_float(metadata.get("age", 0))
    male = 1.0 if str(metadata.get("gender", "")).lower().startswith("m") else 0.0

    risk_score = 0.0
    if glucose >= 200:
        risk_score += 2.0
    elif glucose >= 180:
        risk_score += 1.0
    if systolic >= 160 or diastolic >= 100:
        risk_score += 1.5
    elif systolic >= 140 or diastolic >= 90:
        risk_score += 0.7
    if age >= 65:
        risk_score += 0.6

    # ==============================
    # 4) Combined-probability logic
    # ==============================
    mild_idx = DEFAULT_LABELS.index("Mild")
    mod_idx = DEFAULT_LABELS.index("Moderate")
    sev_idx = DEFAULT_LABELS.index("Severe")
    pdr_idx = DEFAULT_LABELS.index("PDR")
    no_idx = DEFAULT_LABELS.index("No_DR")

    early_prob = float(probs[mild_idx] + probs[mod_idx])
    advanced_prob = float(probs[sev_idx] + probs[pdr_idx])
    no_prob = float(probs[no_idx])

    # --- Thresholds ---
    THRESH_TOP_CONF = 0.45
    THRESH_UNCERTAIN = 0.20
    THRESH_EARLY_COMBINED = 0.45
    THRESH_ADVANCED_COMBINED = 0.45

    # Start
    final_label = merged_label
    conf_pct = float(top_prob * 100)

    # 1) Strong Advanced_DR
    if advanced_prob >= THRESH_ADVANCED_COMBINED or (advanced_prob >= 0.35 and risk_score >= 1.5):
        final_label = "Advanced_DR"
        conf_pct = max(conf_pct, advanced_prob * 100, 60.0)

    # 2) Strong Early_DR
    elif early_prob >= THRESH_EARLY_COMBINED or (early_prob >= 0.30 and risk_score >= 0.7):
        final_label = "Early_DR"
        conf_pct = max(conf_pct, early_prob * 100, 50.0)

    # 3) Confident No_DR
    elif no_prob >= THRESH_TOP_CONF:
        final_label = "No_DR"
        conf_pct = max(conf_pct, no_prob * 100, 40.0)

    # 4) Weak/ambiguous -> use vitals fallback
    else:
        if risk_score >= 2.0:
            final_label = "Advanced_DR"
            conf_pct = max(conf_pct, 65.0)
        elif risk_score >= 0.8:
            final_label = "Early_DR"
            conf_pct = max(conf_pct, 55.0)
        else:
            final_label = "No_DR"
            conf_pct = max(conf_pct, 40.0)

    # 5) Only label AI Uncertain if BOTH image + meta are weak
    if top_prob < THRESH_UNCERTAIN and early_prob < 0.25 and advanced_prob < 0.25 and risk_score < 0.5:
        final_label = "AI Uncertain"
        conf_pct = max(conf_pct, 20.0)

    # Meta-model (optional gentle correction)
    meta_model = PICKLES.get("meta")
    if meta_model is not None and top_prob < 0.65:
        try:
            meta_feat = np.array([[age, male, systolic, diastolic, glucose]])
            # handle xgboost.Booster vs sklearn-like estimators
            meta_pred = None
            try:
                # sklearn-like API
                meta_pred = meta_model.predict(meta_feat)
            except Exception:
                # maybe an xgboost.Booster - convert to DMatrix
                try:
                    import xgboost as xgb
                    dmat = xgb.DMatrix(meta_feat)
                    meta_pred = meta_model.predict(dmat)
                except Exception as ex2:
                    print("Meta model predict failed fallback:", ex2)
            if meta_pred is not None:
                mval = meta_pred[0] if isinstance(meta_pred, (list, np.ndarray)) else meta_pred
                meta_label_raw = DEFAULT_LABELS[int(mval)]
                meta_merged = MERGED_LABELS.get(meta_label_raw, meta_label_raw)
                if meta_merged != final_label and top_prob < 0.5:
                    final_label = meta_merged
                    conf_pct = max(conf_pct, 55.0)
        except Exception as e:
            print("Meta model correction failed:", e)


    # ==============================
    # 6) Grad-CAM Explanation
    # ==============================
    gradcam_url = None

    # ✅ Generate Grad-CAM only for Early_DR and Advanced_DR
    if final_label in ["Early_DR", "Advanced_DR"]:
        try:
            print(f"[INFO] Generating Grad-CAM for {final_label}...")
            # Try Grad-CAM on original image first
            gradcam_url = smooth_gradcam(IMG_MODEL, image_path, top_idx, n_samples=8, stdev_spread=0.08)
            
            # If nothing returned, retry with the enhanced image
            if not gradcam_url:
                print("[WARN] Grad-CAM failed on original, retrying with enhanced image...")
                gradcam_url = smooth_gradcam(IMG_MODEL, enhanced, top_idx, n_samples=8, stdev_spread=0.08)
            
            if gradcam_url:
                print(f"[SUCCESS] Grad-CAM saved: {gradcam_url}")
            else:
                print("[ERROR] Grad-CAM generation returned None even after retry.")
        except Exception as e:
            print("[ERROR] Grad-CAM exception:", e)
            gradcam_url = None
    else:
        # Skip Grad-CAM for healthy or uncertain predictions
        print(f"[INFO] Skipping Grad-CAM for {final_label} (No lesions or uncertain).")
        gradcam_url = None

    # Ensure proper return
    conf_pct = float(min(max(conf_pct, 0.0), 99.0))
    return final_label, conf_pct, gradcam_url



# -------------------------
# PDF report
# -------------------------
def generate_pdf(patient, result, image_path, gradcam_path, out_path):
    c = canvas.Canvas(out_path, pagesize=letter)
    width, height = letter
    y = height - 40
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, "VisionAI - Retinal Screening Report")
    y -= 30
    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 20
    for k in ['name','age','gender','address','contact','systolic','diastolic','glucose']:
        v = patient.get(k, "")
        c.drawString(40, y, f"{k.capitalize()}: {v}")
        y -= 14
    y -= 6
    c.setFont("Helvetica", 11)
    for line in result.split("\n"):
        c.drawString(40, y, line)
        y -= 14
    y -= 10
    try:
        if image_path and os.path.exists(image_path):
            c.drawImage(str(image_path), 40, y-150, width=200, height=150)
        if gradcam_path:
            gpath = BASE_DIR / gradcam_path.lstrip("/")
            if gpath.exists():
                c.drawImage(str(gpath), 260, y-150, width=200, height=150)
    except Exception as e:
        print("PDF image add failed:", e)
    c.showPage(); c.save()

# -------------------------
# Routes
# -------------------------
# -------------------------
# Landing Page + Login
# -------------------------
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/start')
def start():
    return render_template('index.html')

# -------------------------
# Simple login system (for demo)
# -------------------------
app.secret_key = "visionai-secure-key"

# Dummy credentials for review/demo
USERS = {
    "admin": "vision2025",   # username: admin, password: vision2025
    "doctor": "drvision"
}

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if username in USERS and USERS[username] == password:
            session["user"] = username
            flash("Login successful!", "success")
            return redirect(url_for("logs"))
        else:
            flash("Invalid username or password", "danger")
            return redirect(url_for("login"))

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Logged out successfully", "info")
    return redirect(url_for("home"))

# Protect logs page (only accessible after login)
@app.before_request
def restrict_logs():
    protected_routes = ["/logs"]
    if any(request.path.startswith(r) for r in protected_routes):
        if "user" not in session:
            flash("Please login to access logs", "warning")
            return redirect(url_for("login"))




@app.route("/analyze", methods=["POST"])
def analyze():
    data = {k: request.form.get(k, "") for k in ["name","address","contact","age","gender","systolic","diastolic","glucose"]}
    image_file = request.files.get('file')
    camera_data = request.form.get('camera_image')

    if not image_file and not camera_data:
        flash("No image provided", "danger"); return redirect(url_for("index"))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if image_file and image_file.filename:
        fname = f"{Path(image_file.filename).stem}_{ts}{Path(image_file.filename).suffix}"
        image_path = str(UPLOAD_FOLDER / fname)
        image_file.save(image_path)
    else:
        header, encoded = camera_data.split(",", 1)
        image_path = str(UPLOAD_FOLDER / f"camera_{ts}.png")
        with open(image_path, "wb") as f:
            f.write(base64.b64decode(encoded))

    res = predict(image_path, data)
    if res is None:
        flash("Prediction failed — check server logs (see console).", "danger")
        return redirect(url_for("index"))
    label, confidence, gradcam_url = res


    summary = f"Predicted Stage: {label}"
    if safe_float(data.get("glucose", 0)) > 180:
        summary += "\nElevated glucose detected; increases DR risk."
    if safe_float(data.get("systolic", 0)) > 140:
        summary += "\nSystolic BP above 140 — consider BP management."

    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute('''INSERT INTO predictions (name,address,contact,age,gender,systolic,diastolic,glucose,result,confidence,date,image_path)
                 VALUES (?,?,?,?,?,?,?,?,?,?,?,?)''',
              (data['name'],data['address'],data['contact'],data['age'],data['gender'],data['systolic'],data['diastolic'],data['glucose'],label,float(confidence),datetime.now().strftime("%Y-%m-%d %H:%M:%S"),image_path))
    conn.commit(); conn.close()

    pdf_out = REPORT_FOLDER / f"report_{ts}.pdf"
    generate_pdf(data, summary, image_path, gradcam_url, str(pdf_out))

    return render_template("result.html",
                           name=data.get("name",""), age=data.get("age",""), gender=data.get("gender",""),
                           systolic=data.get("systolic",""), diastolic=data.get("diastolic",""), glucose=data.get("glucose",""),
                           result_label=label, confidence=confidence,
                           image_path="/"+os.path.relpath(image_path, BASE_DIR).replace("\\","/"),
                           gradcam_path=gradcam_url,
                           pdf_path="/"+os.path.relpath(pdf_out, BASE_DIR).replace("\\","/"))

@app.route("/logs")
def logs():
    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM predictions ORDER BY date DESC").fetchall()
    conn.close()
    return render_template("logs.html", rows=rows)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
