# utils/pdf_template_utils.py
import os
from datetime import datetime
from textwrap import wrap
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ------------------------------------------------------------------
# 📝 Font setup (Kannada + English with Times Roman)
# ------------------------------------------------------------------
KAN_FONT_PATH = os.path.join("static", "fonts", "NotoSansKannada-Regular.ttf")
KANNADA_FONT_REGISTERED = False
if os.path.exists(KAN_FONT_PATH):
    try:
        pdfmetrics.registerFont(TTFont("NotoSansKannada", KAN_FONT_PATH))
        KANNADA_FONT_REGISTERED = True
        print("✅ Kannada font registered successfully.")
    except Exception as e:
        print("⚠️ Could not register Kannada font:", e)
else:
    print("⚠️ Kannada font not found at", KAN_FONT_PATH)


def _font(pdf, lang, bold=False, size=10):
    """Choose appropriate font based on language (Times for English)."""
    if lang == "kn" and KANNADA_FONT_REGISTERED:
        pdf.setFont("NotoSansKannada", size)
    else:
        try:
            pdf.setFont("Times-Bold" if bold else "Times-Roman", size)
        except Exception:
            pdf.setFont("Helvetica-Bold" if bold else "Helvetica", size)


# ------------------------------------------------------------------
# HEADER
# ------------------------------------------------------------------
def draw_header(pdf, title, color, width=595, height=842, lang="en"):
    """Top bar header with title and timestamp."""
    pdf.setFillColor(color)
    pdf.rect(0, height - 80, width, 72, fill=True, stroke=False)

    pdf.setFillColor(colors.white)
    _font(pdf, lang, bold=True, size=18)
    pdf.drawString(40, height - 52, title)

    _font(pdf, lang, size=8)
    pdf.drawRightString(width - 40, height - 50,
                        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    pdf.setStrokeColor(colors.white)
    pdf.setLineWidth(1)
    pdf.line(40, height - 85, width - 40, height - 85)


# ------------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------------
def draw_footer(pdf, width=595, lang="en"):
    """Footer text centered at bottom."""
    pdf.setFillColor(colors.HexColor("#555555"))
    _font(pdf, lang, bold=False, size=8)
    footer_text = (
        "VisionAI © 2025 | ವಿವರಣೆ ಮತ್ತು ವರದಿ ವ್ಯವಸ್ಥೆ"
        if lang == "kn"
        else "VisionAI © 2025 | Explainability & Reporting System"
    )
    pdf.drawCentredString(width / 2.0, 28, footer_text)
    pdf.setFillColor(colors.black)


# ------------------------------------------------------------------
# METADATA TABLE (stage-color header + bordered grid)
# ------------------------------------------------------------------
def draw_metadata_table(pdf, meta: dict, x, y, col_width=250, row_height=16,
                        lang="en", header_color="#0077b6"):
    """Draws metadata grid styled as table with stage-colored header."""
    # Header bar
    pdf.setFillColor(header_color if isinstance(header_color, colors.Color) else colors.HexColor(header_color))

    pdf.rect(x - 4, y - 4, col_width * 2 + 8, 22, fill=True, stroke=False)

    _font(pdf, lang, bold=True, size=11)
    pdf.setFillColor(colors.white)
    header_text = "ಮೆಟಾಡೇಟಾ ಸಂಗ್ರಹ" if lang == "kn" else "Metadata Snapshot"
    pdf.drawString(x + 6, y, header_text)
    y -= 26

    # Grid body
    _font(pdf, lang, size=9)
    pdf.setStrokeColor(colors.HexColor("#cfe7f3"))
    pdf.setLineWidth(0.5)

    keys = [
        ("Full_Name", "Name" if lang == "en" else "ಹೆಸರು"),
        ("Age", "Age" if lang == "en" else "ವಯಸ್ಸು"),
        ("Gender", "Gender" if lang == "en" else "ಲಿಂಗ"),
        ("Systolic", "Systolic (mmHg)" if lang == "en" else "ಸಿಸ್ಟಾಲಿಕ್ (mmHg)"),
        ("Diastolic", "Diastolic (mmHg)" if lang == "en" else "ಡಯಾಸ್ಟಾಲಿಕ್ (mmHg)"),
        ("BMI", "BMI"),
        ("Glucose_Level", "Glucose" if lang == "en" else "ಗ್ಲೂಕೋಸ್"),
        ("HbA1c", "HbA1c"),
        ("Cholesterol", "Cholesterol" if lang == "en" else "ಕೊಲೆಸ್ಟ್ರಾಲ್"),
        ("Smoking", "Smoking" if lang == "en" else "ಧೂಮಪಾನ"),
        ("Hypertension", "Hypertension" if lang == "en" else "ರಕ್ತದೊತ್ತಡ"),
        ("Duration", "Diabetes Duration" if lang == "en" else "ಮಧುಮೇಹ ಅವಧಿ"),
    ]

    for k, label in keys:
        val = str(meta.get(k, "N/A"))
        pdf.setStrokeColor(colors.HexColor("#d9edf7"))
        pdf.setFillColor(colors.white)
        pdf.rect(x - 4, y - 2, col_width * 2 + 8, row_height + 4, fill=1, stroke=1)

        pdf.setFillColor(colors.black)
        if lang == "kn" and KANNADA_FONT_REGISTERED:
            pdf.setFont("NotoSansKannada", 10)
        else:
            pdf.setFont("Times-Roman", 9)

        pdf.drawString(x + 6, y + 4, f"{label}: {val}")
        y -= row_height + 3

    return y - 12


# ------------------------------------------------------------------
# IMAGE ROW
# ------------------------------------------------------------------
def draw_image_row(pdf, images: list, x, y, max_w=120, gap=12, max_per_row=4):
    """Draws flexible grid of images (original + explainability)."""
    cur_x = x
    max_h = 0
    drawn = 0

    for img_path in images:
        if not img_path or not os.path.exists(img_path):
            continue
        try:
            img = ImageReader(img_path)
            iw, ih = img.getSize()
            scale = min(max_w / iw, 100 / ih)
            w, h = iw * scale, ih * scale

            if drawn and drawn % max_per_row == 0:
                y -= (max_h + 20)
                cur_x = x
                max_h = 0

            pdf.drawImage(img, cur_x, y - h, width=w, height=h, preserveAspectRatio=True, mask="auto")
            cur_x += w + gap
            max_h = max(max_h, h)
            drawn += 1
        except Exception:
            continue

    return y - (max_h + 30 if drawn else 100)


# ------------------------------------------------------------------
# BULLET / TEXT SECTION (with margins + Kannada spacing)
# ------------------------------------------------------------------
def draw_bullet_section(pdf, x, y, heading, bullets_or_text,
                        heading_color=colors.HexColor("#0077b6"), lang="en"):
    """Draws a heading and bullet/paragraph section with smart spacing."""
    if not bullets_or_text:
        return y - 10

    # Top margin before section
    y -= 14

    # Section heading
    _font(pdf, lang, bold=True, size=12)
    pdf.setFillColor(heading_color)
    pdf.drawString(x, y, heading)
    y -= 18

    # Body
    _font(pdf, lang, bold=False, size=10)
    pdf.setFillColor(colors.black)
    if isinstance(bullets_or_text, str):
        bullets = [b.strip() for b in bullets_or_text.split("\n") if b.strip()]
    else:
        bullets = [b.strip() for b in bullets_or_text if b.strip()]

    bullet_indent = x + 14
    line_height = 15 if lang == "en" else 18

    for b in bullets:
        wrapped = wrap(b, 95 if lang == "en" else 75)

        # Bullet
        pdf.setFont("Times-Bold", 11)
        pdf.drawString(bullet_indent - 10, y, u"\u2022")

        # Text
        text = pdf.beginText()
        text.setTextOrigin(bullet_indent + 4, y)
        if lang == "kn" and KANNADA_FONT_REGISTERED:
            text.setFont("NotoSansKannada", 11)
            text.setCharSpace(0.5)
        else:
            text.setFont("Times-Roman", 10)
        text.textLine(wrapped[0])
        pdf.drawText(text)
        y -= line_height

        # Remaining lines
        for line in wrapped[1:]:
            text = pdf.beginText()
            text.setTextOrigin(bullet_indent + 10, y)
            if lang == "kn" and KANNADA_FONT_REGISTERED:
                text.setFont("NotoSansKannada", 11)
                text.setCharSpace(0.5)
            else:
                text.setFont("Times-Roman", 10)
            text.textLine(line)
            pdf.drawText(text)
            y -= line_height

        y -= 4

        # Page break safety
        if y < 90:
            pdf.showPage()
            draw_header(pdf, heading, heading_color, lang=lang)
            y = 760
            _font(pdf, lang, size=10)
            pdf.setFillColor(colors.black)

    # Bottom margin
    return y - 14
