from deep_translator import GoogleTranslator

def translate_to_kannada(text):
    try:
        translated = GoogleTranslator(source="en", target="kn").translate(text)
        return translated
    except Exception as e:
        return f"[Translation Error] {e}"

if __name__ == "__main__":
    english_text = "This image shows signs of severe diabetic retinopathy."
    print("🌍 English:", english_text)
    kannada_text = translate_to_kannada(english_text)
    print("✅ Kannada Translation:", kannada_text)
