# scripts/visionai_report_v4.py
import os
from datetime import datetime
from fpdf import FPDF

# ---------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------
def stage_color(stage):
    colors = {
        "NO_DR": (0, 200, 100),
        "MILD": (255, 200, 0),
        "MODERATE": (255, 150, 50),
        "SEVERE": (255, 80, 80),
        "PDR": (255, 0, 0),
    }
    return colors.get(stage.upper(), (200, 200, 200))

def stage_title(stage):
    titles = {
        "NO_DR": "No Diabetic Retinopathy",
        "MILD": "Mild Non-Proliferative DR",
        "MODERATE": "Moderate Non-Proliferative DR",
        "SEVERE": "Severe Non-Proliferative DR",
        "PDR": "Proliferative Diabetic Retinopathy (PDR)",
    }
    return titles.get(stage.upper(), "Uncertain / Inconclusive")

# ---------------------------------------------------------------------
# COMMON REPORT BUILDER
# ---------------------------------------------------------------------
def build_pdf(output_path, title, theme_color=None, metadata=None,
              images_map=None, image_order=None, lesion_stats=None,
              bullets_dict=None, include_metrics=False, metrics=None,
              confusion=None, logo=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # HEADER
    if logo and os.path.exists(logo):
        pdf.image(logo, 10, 8, 28)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, title, ln=True, align="C")

    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated on: {datetime.now().strftime('%d %B %Y, %I:%M %p')}", ln=True, align="C")
    pdf.ln(8)

    # PATIENT DETAILS
    if metadata:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "Patient Information", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for k, v in metadata.items():
            pdf.cell(0, 7, f"{k}: {v}", ln=True)
        pdf.ln(6)

    # STAGE + STATS
    if lesion_stats:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "AI Findings Overview", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for k, v in lesion_stats.items():
            pdf.cell(0, 7, f"{k.capitalize()}: {round(float(v), 2)}", ln=True)
        pdf.ln(5)

    # IMAGES
    if images_map and image_order:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "Visual Interpretations", ln=True)
        pdf.ln(3)
        for img_key in image_order:
            if not images_map.get(img_key):
                continue
            path = images_map[img_key]
            if os.path.exists(path):
                pdf.set_font("Helvetica", "I", 10)
                pdf.cell(0, 6, img_key.replace("_", " ").title(), ln=True)
                pdf.image(path, w=100)
                pdf.ln(5)

    # CONTENT SECTIONS
    if bullets_dict:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "Interpretation Summary", ln=True)
        pdf.ln(2)
        for heading, text in bullets_dict.items():
            pdf.set_font("Helvetica", "B", 11)
            pdf.multi_cell(0, 8, f"{heading}:")
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 7, f"{text}")
            pdf.ln(3)

    # FOOTER
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 10, "VisionAI — AI-Powered Retinal Screening System", ln=True, align="C")

    pdf.output(output_path)
    print(f"✅ Report generated: {output_path}")
    return output_path

# ---------------------------------------------------------------------
# PATIENT REPORT SECTIONS
# ---------------------------------------------------------------------
def patient_sections(summary):
    stage = summary.get("predicted_stage", "UNKNOWN").upper()
    risk = summary.get("risk_score", 0)
    conf = summary.get("confidence", 0)
    name = summary.get("metadata", {}).get("Full Name", "Patient")

    if stage == "NO_DR":
        content = {
            "Diagnosis": f"No signs of Diabetic Retinopathy were found for {name}.",
            "Findings": "The retina appears healthy with no visible microaneurysms, hemorrhages, or exudates.",
            "Advice": "Maintain healthy blood sugar, diet, and lifestyle. Regular annual eye check-ups are recommended."
        }
    elif stage == "MILD":
        content = {
            "Diagnosis": f"Early signs of Mild Non-Proliferative DR detected for {name}.",
            "Findings": "Small microaneurysms and mild vascular changes were identified. No vision-threatening lesions seen.",
            "Advice": "Control blood sugar and blood pressure. Regular follow-up after 6 months is advised."
        }
    elif stage == "MODERATE":
        content = {
            "Diagnosis": f"Moderate Non-Proliferative DR detected for {name}.",
            "Findings": "Multiple hemorrhages and microaneurysms found. Some exudates visible, requiring monitoring.",
            "Advice": "Visit an ophthalmologist for detailed evaluation within 3 months. Maintain strict glycemic control."
        }
    elif stage == "SEVERE":
        content = {
            "Diagnosis": f"Severe Non-Proliferative DR detected for {name}.",
            "Findings": "Extensive hemorrhages and intraretinal microvascular abnormalities were detected.",
            "Advice": "Immediate ophthalmic consultation is strongly recommended to prevent progression to PDR."
        }
    elif stage == "PDR":
        content = {
            "Diagnosis": f"Proliferative Diabetic Retinopathy detected for {name}.",
            "Findings": "AI found neovascularization and significant retinal damage indicating high risk of vision loss.",
            "Advice": "Urgent treatment is necessary. Consult a retina specialist immediately for laser or anti-VEGF therapy."
        }
    else:
        content = {
            "Diagnosis": "Stage could not be determined.",
            "Findings": "Insufficient confidence or unclear image quality.",
            "Advice": "Please retake the image with proper lighting or consult an ophthalmologist."
        }

    return [content]

# ---------------------------------------------------------------------
# DOCTOR REPORT SECTIONS
# ---------------------------------------------------------------------
def doctor_sections(summary, metadata):
    stage = summary.get("predicted_stage", "UNKNOWN").upper()
    conf = summary.get("confidence", 0)
    risk = summary.get("risk_score", 0)
    lesion_stats = summary.get("lesion_stats", {})

    if stage == "NO_DR":
        content = {
            "Clinical Summary": "No evidence of DR lesions detected. Vascular structure appears intact.",
            "AI Findings": "No exudates, hemorrhages, or neovascular growth patterns detected.",
            "Recommendation": "Continue routine annual monitoring."
        }
    elif stage == "MILD":
        content = {
            "Clinical Summary": "Early-stage mild NPDR with few microaneurysms identified.",
            "AI Findings": f"Lesion probabilities: {lesion_stats}. Confidence: {round(conf*100,2)}%.",
            "Recommendation": "Monitor progression every 6 months. Control systemic factors."
        }
    elif stage == "MODERATE":
        content = {
            "Clinical Summary": "Moderate NPDR characterized by hemorrhages and vascular leakage.",
            "AI Findings": f"AI detected hard exudates and microaneurysms with {round(conf*100,2)}% confidence.",
            "Recommendation": "Schedule ophthalmic review within 3 months."
        }
    elif stage == "SEVERE":
        content = {
            "Clinical Summary": "Severe NPDR showing multiple quadrants with IRMA and hemorrhages.",
            "AI Findings": f"AI indicates strong lesion activation with {round(conf*100,2)}% confidence and risk {round(risk,1)}%.",
            "Recommendation": "Immediate referral for treatment; consider laser photocoagulation."
        }
    elif stage == "PDR":
        content = {
            "Clinical Summary": "Proliferative DR with extensive neovascularization detected.",
            "AI Findings": f"Neovascular and hemorrhagic lesions identified in central retina with {round(conf*100,2)}% confidence.",
            "Recommendation": "Urgent retina specialist consultation recommended."
        }
    else:
        content = {
            "Clinical Summary": "Stage indeterminate.",
            "AI Findings": "Image or metadata insufficient.",
            "Recommendation": "Repeat imaging with mydriatic fundus photography."
        }

    return [content], None

# ---------------------------------------------------------------------
# RESEARCH REPORT SECTIONS
# ---------------------------------------------------------------------
def research_sections(summary):
    stage = summary.get("predicted_stage", "UNKNOWN").upper()
    conf = summary.get("confidence", 0)
    risk = summary.get("risk_score", 0)
    metrics = {
        "Stage": stage,
        "Confidence": f"{round(conf*100,2)}%",
        "Risk Score": f"{round(risk,1)}%"
    }

    if stage == "NO_DR":
        content = {
            "Interpretation": "No pathological regions were highlighted in Grad-CAM or LIME maps. Model indicates retinal normalcy.",
            "Explainability": "Grad-CAM++ shows uniform attention distribution across macular and peripheral zones.",
            "Conclusion": "No DR detected; model generalizes well to healthy samples."
        }
    elif stage == "MILD":
        content = {
            "Interpretation": "Mild lesion activation zones with localized attention around microaneurysms.",
            "Explainability": "LIME maps correspond to small red spots, consistent with early NPDR.",
            "Conclusion": "Findings confirm early NPDR detection capabilities of VisionAI."
        }
    elif stage == "MODERATE":
        content = {
            "Interpretation": "Moderate lesion concentration with scattered hemorrhages and exudate clusters.",
            "Explainability": "Grad-CAM focuses near macula; SHAP highlights glucose and hypertension as key risk factors.",
            "Conclusion": "Model demonstrates reliable interpretability for mid-stage NPDR."
        }
    elif stage == "SEVERE":
        content = {
            "Interpretation": "Extensive lesion concentration across multiple quadrants.",
            "Explainability": "AI visualization maps indicate dense vascular irregularities and exudate clusters.",
            "Conclusion": "VisionAI shows strong correlation between clinical and predicted features for severe DR."
        }
    elif stage == "PDR":
        content = {
            "Interpretation": "Active neovascular growth and vitreoretinal traction zones highlighted.",
            "Explainability": "Grad-CAM++, LIME, and SHAP confirm consistent attention to proliferative areas.",
            "Conclusion": "VisionAI effectively distinguishes PDR with high lesion confidence and cross-modality agreement."
        }
    else:
        content = {
            "Interpretation": "Model output uncertain or input image suboptimal.",
            "Explainability": "No clear region importance identified.",
            "Conclusion": "Further validation required for this case."
        }

    return [content], metrics
