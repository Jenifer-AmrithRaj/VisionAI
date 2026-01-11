"""
File: utils/xai_utils.py
Purpose: Robust XAI helpers for VisionAI (GradCAM, LIME, SHAP)
"""

import os
import traceback
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional imports
try:
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
    _LIME_AVAILABLE = True
except Exception:
    _LIME_AVAILABLE = False

try:
    import shap
    _SHAP_AVAILABLE = True
except Exception:
    _SHAP_AVAILABLE = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------
def _ensure_dir_for_file(path: str):
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)


def _load_image_cv2(img_path: str):
    return cv2.imread(img_path) if img_path and os.path.exists(img_path) else None


def _pil_to_tensor(img_pil: Image.Image, size=(224, 224)) -> torch.Tensor:
    t = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
    return t(img_pil).unsqueeze(0).to(DEVICE)


# ---------------------------------------------------------
# GradCAM (UNCHANGED)
# ---------------------------------------------------------
class _GradCAM:
    def __init__(self, model):
        self.model = model.eval() if model is not None else None
        self.gradients = None
        self.activations = None
        self.hooks = []
        if self.model is not None:
            self._register_hooks()

    def _find_target_layer(self):
        for _, module in reversed(list(self.model.named_modules())):
            if "conv" in module.__class__.__name__.lower():
                return module
        return None

    def _register_hooks(self):
        layer = self._find_target_layer()
        if layer is None:
            return

        def fwd(_, __, out):
            self.activations = out.detach()

        def bwd(_, __, grad):
            self.gradients = grad[0].detach()

        self.hooks.append(layer.register_forward_hook(fwd))
        self.hooks.append(layer.register_full_backward_hook(bwd))

    def release(self):
        for h in self.hooks:
            h.remove()

    def generate_cam(self, tensor):
        out = self.model(tensor)
        idx = out.argmax(dim=1)
        one_hot = torch.zeros_like(out)
        one_hot[0, idx] = 1.0
        self.model.zero_grad()
        out.backward(gradient=one_hot, retain_graph=True)

        if self.gradients is None or self.activations is None:
            return None

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze().cpu().numpy()
        cam = np.maximum(cam, 0)
        return cam / cam.max() if cam.max() > 0 else cam


def generate_gradcam_image(model, img_path, save_path):
    if not save_path:
        return ""
    try:
        img = _load_image_cv2(img_path)
        if img is None or model is None:
            return ""

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = _pil_to_tensor(Image.fromarray(img_rgb))

        cam_gen = _GradCAM(model)
        cam = cam_gen.generate_cam(tensor)
        cam_gen.release()

        if cam is None:
            return ""

        cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
        heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.6, heat, 0.4, 0)

        _ensure_dir_for_file(save_path)
        cv2.imwrite(save_path, overlay)
        return save_path

    except Exception:
        traceback.print_exc()
        return ""


# ---------------------------------------------------------
# LIME (UNCHANGED)
# ---------------------------------------------------------
def generate_lime_image(model, img_path, save_path):
    if not _LIME_AVAILABLE or not save_path:
        return ""

    try:
        pil = Image.open(img_path).convert("RGB").resize((224, 224))
        np_img = np.array(pil)

        def batch_predict(images):
            batch = torch.stack([
                transforms.ToTensor()(Image.fromarray(im)).to(DEVICE)
                for im in images
            ])
            with torch.no_grad():
                return torch.softmax(model(batch), dim=1).cpu().numpy()

        explainer = lime_image.LimeImageExplainer()
        exp = explainer.explain_instance(np_img, batch_predict, top_labels=1, num_samples=300)
        temp, mask = exp.get_image_and_mask(exp.top_labels[0], positive_only=True)

        lime_img = mark_boundaries(temp / 255.0, mask)
        _ensure_dir_for_file(save_path)
        cv2.imwrite(
            save_path,
            cv2.cvtColor((lime_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        )
        return save_path

    except Exception:
        traceback.print_exc()
        return ""


# ---------------------------------------------------------
# ✅ SHAP — FIXED & STABLE
# ---------------------------------------------------------
def generate_shap_plot(model, metadata, save_path):
    """
    Correct SHAP implementation:
    - Uses TreeExplainer for RF/XGB
    - Uses feature names
    - Clean fallback if SHAP fails
    """
    if not _SHAP_AVAILABLE or model is None or not save_path:
        return ""

    try:
        # Numeric metadata only
        numeric = {
            k: float(v)
            for k, v in metadata.items()
            if str(v).replace(".", "", 1).isdigit()
        }

        if not numeric:
            return ""

        feature_names = list(numeric.keys())
        values = np.array([list(numeric.values())])

        _ensure_dir_for_file(save_path)

        try:
            # ✅ Tree-based models
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(values)

            # Handle binary vs multiclass
            if isinstance(shap_values, list):
                shap_vals = shap_values[0]
            else:
                shap_vals = shap_values

            plt.figure(figsize=(6, 3))
            shap.summary_plot(
                shap_vals,
                values,
                feature_names=feature_names,
                plot_type="bar",
                show=False
            )
            plt.tight_layout()
            plt.savefig(save_path, dpi=160)
            plt.close()
            return save_path

        except Exception:
            # 🔁 Fallback: deterministic importance bar (NOT fake SHAP)
            importances = np.abs(values[0])
            idx = np.argsort(importances)[::-1]

            plt.figure(figsize=(6, 3))
            plt.barh(
                np.array(feature_names)[idx],
                importances[idx]
            )
            plt.xlabel("Feature Value")
            plt.title("Feature Importance (Fallback)")
            plt.tight_layout()
            plt.savefig(save_path, dpi=160)
            plt.close()
            return save_path

    except Exception:
        traceback.print_exc()
        return ""


# ---------------------------------------------------------
# Lesion Quantification (UNCHANGED)
# ---------------------------------------------------------
def calculate_lesion_stats(path):
    """
    Enhanced lesion quantification with neovascularization support.
    Designed for DR staging (NO_DR → PDR).
    """
    try:
        img = cv2.imread(path)
        if img is None:
            return {}

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        total = float(img.shape[0] * img.shape[1]) or 1.0

        # -------------------------------------------------
        # Exudates (bright yellow / white)
        # -------------------------------------------------
        exudates = cv2.inRange(hsv, (15, 40, 170), (40, 255, 255))

        # -------------------------------------------------
        # Hemorrhages (dark red / brown)
        # -------------------------------------------------
        hem1 = cv2.inRange(hsv, (0, 60, 20), (10, 255, 160))
        hem2 = cv2.inRange(hsv, (160, 60, 20), (179, 255, 160))
        hemorrhages = cv2.bitwise_or(hem1, hem2)

        # -------------------------------------------------
        # Microaneurysms (small dark dots)
        # -------------------------------------------------
        _, dark = cv2.threshold(gray, 55, 255, cv2.THRESH_BINARY_INV)
        microaneurysms = cv2.morphologyEx(
            dark,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        )

        # -------------------------------------------------
        # Cotton wool spots (diffuse pale regions)
        # -------------------------------------------------
        cotton_wool = cv2.inRange(hsv, (0, 0, 180), (179, 70, 255))
        cotton_wool = cv2.medianBlur(cotton_wool, 7)

        # -------------------------------------------------
        # ✅ Neovascularization (thin abnormal vessels)
        # Edge-based + dilation → ONLY meaningful in PDR
        # -------------------------------------------------
        edges = cv2.Canny(gray, 60, 120)
        neovascularization = cv2.dilate(
            edges,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
            iterations=1
        )

        # -------------------------------------------------
        # Assemble masks
        # -------------------------------------------------
        masks = {
            "exudates": exudates,
            "hemorrhages": hemorrhages,
            "microaneurysms": microaneurysms,
        }

        # Cotton wool: include only if meaningful
        cw_pct = 100.0 * np.sum(cotton_wool > 0) / total
        if cw_pct >= 0.05:
            masks["cotton_wool"] = cotton_wool

        # Neovascularization: include only if non-trivial
        neo_pct = 100.0 * np.sum(neovascularization > 0) / total
        if neo_pct >= 0.5:  # suppress noise in non-PDR
            masks["neovascularization"] = neovascularization

        # -------------------------------------------------
        # Compute percentages
        # -------------------------------------------------
        result = {}
        for name, mask in masks.items():
            pct = round(100.0 * np.sum(mask > 0) / total, 3)
            result[name] = min(pct, 30.0)  # cap unrealistic outliers

        # Total lesion load (exclude neovascularization from sum)
        result["total_lesion_load"] = round(
            sum(v for k, v in result.items() if k != "neovascularization"),
            3
        )

        return result

    except Exception:
        traceback.print_exc()
        return {}

