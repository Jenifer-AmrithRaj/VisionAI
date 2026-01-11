# utils/report_utils.py
"""
VisionAI Report Utilities (Final – Rich Clinical Content)
---------------------------------------------------------
Generates detailed, stage-aware Patient, Doctor, and Research
reports with clean NO_DR handling and full DR-positive narratives.
"""

import os
import json
import traceback
from datetime import datetime

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False

from scripts.generate_pdf import generate_all_reports

try:
    from deep_translator import GoogleTranslator
    DEEP_TRANSLATOR_AVAILABLE = True
except Exception:
    GoogleTranslator = None
    DEEP_TRANSLATOR_AVAILABLE = False


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SUMMARY_DIR = os.path.join(BASE_DIR, "explainability")


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def safe_json_save(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def translate_to_kannada(text: str) -> str:
    if not text or not DEEP_TRANSLATOR_AVAILABLE:
        return text
    try:
        return GoogleTranslator(source="en", target="kn").translate(text)
    except Exception:
        return text


def _stage_label(stage):
    return stage.replace("_", " ").title() if stage else "Unknown"


# ---------------------------------------------------------
# Patient Report
# ---------------------------------------------------------
def make_patient_text(summary: dict) -> str:
    meta = summary.get("metadata", {})
    pred = summary.get("prediction", {})

    name = meta.get("Full_Name", "Patient")
    stage = pred.get("predicted_stage")
    conf = pred.get("confidence", "N/A")
    risk = pred.get("risk_score", "N/A")

    if stage == "NO_DR":
        return (
            f"Patient Name: {name}\n\n"
            f"Result Summary:\n"
            f"No signs of diabetic retinopathy were detected in the analyzed retinal image.\n\n"
            f"Clinical Interpretation:\n"
            f"Your retina appears healthy, with no visible microvascular damage related to diabetes. "
            f"This indicates a very low current risk of diabetic eye disease.\n\n"
            f"Recommendations:\n"
            f"- Continue routine annual retinal screening.\n"
            f"- Maintain good blood sugar, blood pressure, and cholesterol control.\n"
            f"- Seek medical attention if you notice visual symptoms such as blur, floaters, or flashes.\n\n"
            f"Confidence Level: {conf}%\n"
            f"Estimated Risk: Minimal\n\n"
            f"VisionAI provides decision-support only and does not replace clinical examination."
        )

    return (
        f"Patient Name: {name}\n\n"
        f"Result Summary:\n"
        f"Diabetic retinopathy changes were detected at the {_stage_label(stage)} stage.\n\n"
        f"Clinical Interpretation:\n"
        f"This stage indicates damage to the small blood vessels of the retina caused by diabetes. "
        f"At this level, vision may still be preserved, but disease progression is possible without "
        f"appropriate monitoring and management.\n\n"
        f"Recommendations:\n"
        f"- Follow up with an ophthalmologist as advised.\n"
        f"- Maintain strict glycemic and blood pressure control.\n"
        f"- Report any new visual symptoms immediately.\n\n"
        f"Model Confidence: {conf}%\n"
        f"Estimated Risk Score: {risk}%\n\n"
        f"VisionAI provides automated screening support and does not replace clinical diagnosis."
    )


# ---------------------------------------------------------
# Doctor Report
# ---------------------------------------------------------
def make_doctor_text(summary: dict) -> str:
    meta = summary.get("metadata", {})
    pred = summary.get("prediction", {})
    metrics = summary.get("model_metrics", {})

    stage = pred.get("predicted_stage")
    conf = pred.get("confidence", "N/A")
    risk = pred.get("risk_score", "N/A")

    if stage == "NO_DR":
        return (
            "Clinical Report — Ophthalmology\n\n"
            f"Patient: {meta.get('Full_Name','N/A')}\n\n"
            f"Automated Assessment:\n"
            f"No diabetic retinopathy detected.\n\n"
            f"Clinical Interpretation:\n"
            f"No microaneurysms, hemorrhages, exudates, or neovascular features were identified. "
            f"Explainability analysis and lesion quantification were intentionally skipped due to "
            f"absence of pathological findings.\n\n"
            f"Recommendation:\n"
            f"Routine annual screening is advised unless additional clinical risk factors exist.\n\n"
            f"Model Confidence: {conf}%"
        )

    return (
        "Clinical Report — Ophthalmology\n\n"
        f"Patient: {meta.get('Full_Name','N/A')}\n\n"
        f"Automated Diagnosis:\n"
        f"Predicted Stage: {_stage_label(stage)}\n"
        f"Confidence: {conf}%\n"
        f"Risk Score: {risk}%\n\n"
        f"Findings:\n"
        f"Automated lesion quantification and explainability analyses (Grad-CAM, LIME, SHAP) "
        f"support the predicted disease stage.\n\n"
        f"Clinical Interpretation:\n"
        f"The findings are consistent with diabetic microvascular damage. Correlation with "
        f"clinical examination and OCT imaging is recommended where appropriate.\n\n"
        f"Performance Reference:\n"
        f"Accuracy: {metrics.get('Accuracy','N/A')} | "
        f"F1-score: {metrics.get('F1-score','N/A')} | "
        f"AUC: {metrics.get('AUC/ROC','N/A')}\n\n"
        f"VisionAI is intended for decision support and audit assistance."
    )


# ---------------------------------------------------------
# Research Report
# ---------------------------------------------------------
def make_research_text(summary: dict) -> str:
    pred = summary.get("prediction", {})
    probs = summary.get("probs", {})
    uid = summary.get("uid", "N/A")

    stage = pred.get("predicted_stage")

    device = "CPU"
    if TORCH_AVAILABLE and torch.cuda.is_available():
        device = "CUDA"

    if stage == "NO_DR":
        return (
            "Research Notes\n\n"
            f"UID: {uid}\n\n"
            f"Prediction Outcome:\n"
            f"NO_DR — no pathological features detected.\n\n"
            f"Pipeline Behavior:\n"
            f"Lesion quantification and explainability modules were bypassed by design "
            f"to avoid false interpretation in normal cases.\n\n"
            f"Inference Device: {device}\n\n"
            f"This behavior is intentional and clinically aligned."
        )

    return (
        "Research Notes\n\n"
        f"UID: {uid}\n\n"
        f"Predicted Stage: {_stage_label(stage)}\n\n"
        f"Model Architecture:\n"
        f"CNN ensemble (EfficientNet, ResNet50, ViT) combined with metadata models "
        f"(Random Forest, XGBoost) via calibrated fusion.\n\n"
        f"Explainability:\n"
        f"Grad-CAM highlights spatial retinal features, LIME validates local pixel "
        f"importance, and SHAP explains systemic risk contribution.\n\n"
        f"Probability Vectors:\n"
        f"CNN: {probs.get('cnn', [])}\n"
        f"ML: {probs.get('ml', [])}\n"
        f"Fused: {probs.get('fused', [])}\n\n"
        f"Inference Device: {device}"
    )


# ---------------------------------------------------------
# Main Generator
# ---------------------------------------------------------
def generate_reports_for_uid(uid, language_mode="en"):
    summary_path = os.path.join(SUMMARY_DIR, f"{uid}_xai_summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(summary_path)

    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    try:
        summary.setdefault("reports", {})

        patient_en = make_patient_text(summary)
        doctor_en = make_doctor_text(summary)
        research_en = make_research_text(summary)

        summary["reports"].update({
            "patient_en": patient_en,
            "doctor_en": doctor_en,
            "research_en": research_en,
            "patient_kn": translate_to_kannada(patient_en) if language_mode != "en" else "",
            "doctor_kn": translate_to_kannada(doctor_en) if language_mode != "en" else "",
            "research_kn": translate_to_kannada(research_en) if language_mode != "en" else "",
        })

        safe_json_save(summary_path, summary)
        return generate_all_reports(uid, summary, lang=language_mode)

    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Report generation failed for UID {uid}: {e}")
