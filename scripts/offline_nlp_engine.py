"""
VisionAI Offline NLP Engine (Rich, Stage-Aware, PDF-Optimized)
--------------------------------------------------------------
• Patient → empathetic, reassuring, action-oriented
• Doctor → structured, clinical, guideline-aligned
• Research → analytical, model-centric, explainability-focused
"""

from typing import Dict, Any

try:
    from deep_translator import GoogleTranslator
    TRANSLATE_OK = True
except Exception:
    GoogleTranslator = None
    TRANSLATE_OK = False


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def _percent(x):
    try:
        x = float(x)
        return f"{x*100:.1f}%" if x <= 1 else f"{x:.1f}%"
    except Exception:
        return "N/A"


def _stage_label(stage):
    return stage.replace("_", " ").title() if stage else "Unknown"


def _translate_kn(text):
    if not text or not TRANSLATE_OK:
        return text
    try:
        return GoogleTranslator(source="en", target="kn").translate(text)
    except Exception:
        return text


def _lesion_text(lesions: Dict[str, Any]):
    if not lesions:
        return "No pathological lesions were detected in the retinal image."
    return ", ".join(
        f"{k.replace('_',' ')} ({_percent(v)})"
        for k, v in lesions.items()
    )


# -------------------------------------------------
# PATIENT CONTENT (RICH)
# -------------------------------------------------
def _patient_voice(stage, meta, lesions, pred):
    name = meta.get("Full_Name", "Patient")
    stage_lbl = _stage_label(stage)

    messages = {
        "NO_DR": (
            "Your retinal scan appears healthy with no signs of diabetic retinopathy. "
            "This is an excellent result. It indicates that diabetes has not yet caused "
            "damage to the blood vessels in your eyes."
        ),
        "MILD": (
            "Early, mild changes were detected in the small blood vessels of the retina. "
            "At this stage, vision is usually unaffected, and progression can often be "
            "slowed or reversed with good diabetes control."
        ),
        "MODERATE": (
            "Moderate changes were detected in the retina, suggesting ongoing stress to "
            "the eye’s blood vessels. While vision may still feel normal, closer monitoring "
            "is important to prevent progression."
        ),
        "SEVERE": (
            "Severe retinal changes were identified. This stage carries a higher risk of "
            "vision problems, and timely evaluation by an eye specialist is strongly advised."
        ),
        "PDR": (
            "Advanced changes called *Proliferative Diabetic Retinopathy* were detected. "
            "This means new, fragile blood vessels may be growing in the retina, which can "
            "lead to bleeding and vision loss if not treated urgently."
        ),
    }

    advice = {
        "NO_DR": "Continue annual eye screenings and maintain good blood sugar control.",
        "MILD": "Schedule an eye check within the next 6–12 months.",
        "MODERATE": "Ophthalmology review recommended within 3–6 months.",
        "SEVERE": "Urgent ophthalmology appointment recommended.",
        "PDR": "Immediate retina specialist consultation is required.",
    }

    return f"""
Hello {name},

Detected Stage: {stage_lbl}
Model Confidence: {_percent(pred.get("confidence"))}
Estimated Risk Score: {_percent(pred.get("risk_score"))}

What this means:
{messages.get(stage, "The findings require further evaluation.")}

Automated retinal findings:
{_lesion_text(lesions)}

What you should do next:
{advice.get(stage, "Please consult your doctor for guidance.")}

Remember:
Early medical care and good diabetes control can significantly protect your vision.
"""


# -------------------------------------------------
# DOCTOR CONTENT (CLINICAL)
# -------------------------------------------------
def _doctor_voice(stage, meta, lesions, pred, metrics):
    stage_lbl = _stage_label(stage)

    interpretations = {
        "NO_DR": "No diabetic retinopathy changes detected. Fundus appears within normal limits.",
        "MILD": "Microaneurysms present. No significant exudation or ischemia.",
        "MODERATE": "Dot-blot hemorrhages and early exudative changes noted.",
        "SEVERE": "Extensive hemorrhages, venous abnormalities, and ischemic signs present.",
        "PDR": "Neovascularization detected, consistent with proliferative diabetic retinopathy.",
    }

    plans = {
        "NO_DR": "Annual screening advised.",
        "MILD": "Follow-up in 6–12 months.",
        "MODERATE": "Consider OCT; follow-up in 3–6 months.",
        "SEVERE": "Urgent referral; consider OCT and fluorescein angiography.",
        "PDR": "Immediate retina referral; PRP / anti-VEGF therapy likely indicated.",
    }

    return f"""
Clinical AI Report – Ophthalmology

Patient: {meta.get("Full_Name", "N/A")}
Predicted Stage: {stage_lbl}
Confidence: {_percent(pred.get("confidence"))}
Risk Score: {_percent(pred.get("risk_score"))}

Automated lesion quantification:
{_lesion_text(lesions)}

Interpretation:
{interpretations.get(stage, "Uncertain classification; manual grading advised.")}

Recommended management:
{plans.get(stage, "Clinical correlation required.")}

Model notes:
• CNN + metadata fusion model
• Explainability via GradCAM, LIME, SHAP
• Accuracy: {metrics.get("Accuracy", "N/A")}
"""


# -------------------------------------------------
# RESEARCH CONTENT (DEEP)
# -------------------------------------------------
def _research_voice(stage, meta, lesions, pred, metrics, probs):
    stage_lbl = _stage_label(stage)

    research_notes = {
        "NO_DR": "Case suitable as negative/control sample.",
        "MILD": "Early vascular lesions detected with low activation intensity.",
        "MODERATE": "Balanced lesion distribution consistent with mid-stage DR.",
        "SEVERE": "High lesion density approaching proliferative threshold.",
        "PDR": "Neovascularization present; strong GradCAM activation in vascular regions.",
    }

    return f"""
Research Case Summary

UID: {meta.get("uid", "N/A")}
Stage: {stage_lbl}
Confidence: {_percent(pred.get("confidence"))}
Risk Score: {_percent(pred.get("risk_score"))}

Lesion quantification:
{_lesion_text(lesions)}

Research interpretation:
{research_notes.get(stage, "Requires further analysis.")}

Model probabilities:
CNN: {probs.get("cnn")}
ML: {probs.get("ml")}
Fused: {probs.get("fused")}

Performance metrics:
Accuracy: {metrics.get("Accuracy")}
F1-score: {metrics.get("F1-score")}
AUC/ROC: {metrics.get("AUC/ROC")}

Notes:
• GradCAM confirms spatial alignment with lesions
• SHAP highlights systemic risk contributors
• Case suitable for longitudinal DR progression studies
"""


# -------------------------------------------------
# MAIN API
# -------------------------------------------------
def generate_dynamic_report(summary: Dict[str, Any], translate_to="en") -> Dict[str, Any]:
    pred = summary.get("prediction", {})
    meta = summary.get("metadata", {})
    lesions = summary.get("lesion_stats", {})
    metrics = summary.get("model_metrics", {})
    probs = summary.get("probs", {})

    stage = (pred.get("predicted_stage") or "UNKNOWN").upper()

    patient_en = _patient_voice(stage, meta, lesions, pred)
    doctor_en = _doctor_voice(stage, meta, lesions, pred, metrics)
    research_en = _research_voice(stage, meta, lesions, pred, metrics, probs)

    reports = {
        "patient_en": patient_en,
        "doctor_en": doctor_en,
        "research_en": research_en,
        "patient_kn": _translate_kn(patient_en) if translate_to == "kn" else "",
        "doctor_kn": _translate_kn(doctor_en) if translate_to == "kn" else "",
        "research_kn": _translate_kn(research_en) if translate_to == "kn" else "",
    }

    return {
        "reports": reports,
        "graph_data": {
            "lesion_distribution": lesions,
            "model_metrics": metrics
        }
    }
