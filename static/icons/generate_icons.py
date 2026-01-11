# static/icons/generate_icons.py  — FINAL FIXED VERSION
from PIL import Image
from pathlib import Path

# Move up two directories from /static/icons/ to reach project root
project_root = Path(__file__).resolve().parents[2]

# ✅ Correct absolute path to your logo
logo = project_root / "static" / "visionai_logo.png"

# Output directory
icons_dir = project_root / "static" / "icons"
icons_dir.mkdir(parents=True, exist_ok=True)

print(f"🔍 Looking for logo at: {logo}")

if not logo.exists():
    print("❌ Logo not found. Check path above.")
else:
    print("✅ Logo found. Generating icons...")
    im = Image.open(logo).convert("RGBA")

    for size, name in [(192, "icon-192x192.png"), (512, "icon-512x512.png")]:
        out = im.resize((size, size), Image.LANCZOS)
        out.save(icons_dir / name)
        print(f"✅ Saved {name} → {icons_dir / name}")

    print("\n🎉 Done! Icons successfully generated.")
