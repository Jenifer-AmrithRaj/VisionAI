<div align="center">

# 👁️‍🗨️ **VisionAI**
### *Intelligent Explainable Diabetic Retinopathy Screening Platform*

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge)
![Flask](https://img.shields.io/badge/Flask-Web_App-black?style=for-the-badge)
![AI](https://img.shields.io/badge/AI-Explainable-success?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Stable-brightgreen?style=for-the-badge)

🚀 **Clinical-grade AI • Explainable • Production-ready • PWA enabled**

</div>

---

## 🧠 What is VisionAI?

**VisionAI** is a **full-stack, explainable AI clinical decision-support system** for  
**automated Diabetic Retinopathy (DR) screening** using:

- 🖼️ Retinal fundus images  
- 📋 Patient clinical metadata  
- 🧠 Multi-model AI fusion  
- 🔍 Explainable AI (XAI)  
- 📄 Automated clinical reports  

It is designed for **clinicians, researchers, and academic review**.

---

## ✨ Key Features

✅ **5-Stage DR Classification**  
`NO_DR → MILD → MODERATE → SEVERE → PDR`

✅ **CNN + Metadata Fusion**
- EfficientNet
- ResNet50
- Vision Transformer (ViT)
- Random Forest
- XGBoost
- Ensemble Stacking

✅ **Explainable AI (XAI)**
- 🔥 Grad-CAM (spatial regions)
- 🧩 LIME (pixel-level explanation)
- 📊 SHAP (metadata feature importance)

✅ **Automated PDF Reports**
- 👤 Patient Report
- 🩺 Doctor Report
- 🔬 Research Report  
*(Stage-colored, dense, professional)*

✅ **Progressive Web App (PWA)**
- Installable
- Offline-friendly
- Clinic-ready

✅ **System Monitoring**
- CPU / RAM / GPU stats
- Live system logs

---

## 🔐 Login Credentials (Demo)

| Field | Value |
|-----|------|
| **Username** | `admin` |
| **Password** | `VisionAI123` |

➡️ Login URL:  
http://127.0.0.1:5000


---

## 🖥️ Application Workflow

### 1️⃣ Login
Secure clinician login to access the system dashboard.

---

### 2️⃣ Dashboard
View:
- 📈 Total screenings
- 📅 Today’s patients
- 🎯 Average confidence
- ⚠️ Average risk score
- 🧪 Model health overview

Quick access:
- ➕ New Screening
- 🕓 Patient History
- 📊 Doctor Dashboard
- ⚙️ System Logs

---

### 3️⃣ New Screening
Enter:
- Patient demographics
- Systemic risk factors (HbA1c, BP, BMI, duration, etc.)
- Upload or capture **fundus image**

🧠 **Why metadata?**  
Systemic factors significantly improve prediction accuracy and SHAP explainability.

---

### 4️⃣ AI Prediction Pipeline

1. Image preprocessing  
2. CNN inference  
3. Metadata ML inference  
4. Probability fusion  
5. Risk scoring  
6. XAI generation  
7. Lesion quantification  
8. Report generation  

⚡ Runs asynchronously in the background.

---

### 5️⃣ Results Page

Displays:
- 🧠 Predicted DR stage
- 📊 Confidence & risk score
- 📈 Probability distribution
- 🔥 Grad-CAM
- 🧩 LIME
- 📊 SHAP
- 🩺 Lesion statistics
- 📋 Patient metadata

---

## 🔍 Explainable AI (XAI)

| Method | Purpose |
|------|--------|
| **Grad-CAM** | Highlights retinal regions influencing prediction |
| **LIME** | Local pixel-level explanation |
| **SHAP** | Feature importance from metadata |

🟢 **NO_DR cases**  
XAI and lesion analysis are intentionally skipped to prevent misleading interpretation.

---

## 🩺 Lesion Quantification

Automatically detects:
- Microaneurysms
- Exudates
- Hemorrhages
- Cotton-wool spots
- Neovascularization (PDR)

Used in **clinical reasoning & reports**.

---

## 📄 Reports (PDF)

Generated automatically:
- 🧑‍⚕️ Patient-friendly report
- 🩺 Clinician-grade report
- 🔬 Research-grade report

🎨 **Stage-based color coding**
- PDR → Maroon / Red
- Severe → Red
- Moderate → Orange
- Mild → Blue
- NO_DR → Green

Supports **English & Kannada**.

---

## 🌐 Progressive Web App (PWA)

Click **“Install VisionAI”** on the dashboard to:
- Install as native-like app
- Enable offline access
- Use in low-connectivity clinics

---

## ⚙️ System Logs & Monitoring

### 🧠 Resource Monitor
- CPU usage
- RAM usage
- GPU memory & temperature

### 📜 System Logs
- Background tasks
- Model execution
- Report generation
- Errors & warnings

---

## 🗂️ Project Structure

dr_vision_ai/
│
├── app.py                     # Main Flask backend
├── utils/
│   ├── model_utils.py
│   ├── xai_utils.py
│   ├── report_utils.py
│   ├── logger.py
│
├── scripts/
│   ├── generate_pdf.py
│   ├── offline_nlp_engine.py
│
├── templates/                 # HTML pages
├── static/                    # CSS, JS, images, PWA files
├── explainability/            # JSON summaries
├── reports/                   # Generated PDFs
├── uploads/                   # Uploaded images
├── logs/                      # System logs




---

## 🛠️ Installation & Setup

```bash
git clone <repo-url>
cd dr_vision_ai
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py

server: 
http://127.0.0.1:5000


🚀 Deployment Ready

✔️ Local server
✔️ Hospital intranet
✔️ Cloud hosting (Render / AWS / Azure / GCP)

Production tips:

Disable debug

Use Gunicorn / Waitress

Enable HTTPS

Secure credentials via env variables

⚠️ Medical Disclaimer

VisionAI is a clinical decision-support tool.
It does not replace professional ophthalmologist diagnosis.

👨‍⚕️ Ideal For

Academic evaluation & reviews

Explainable medical AI research

Clinical AI demonstrations

Healthcare software deployment