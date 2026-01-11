# utils/preprocess.py
from PIL import Image, ImageOps
import cv2
import numpy as np
import torchvision.transforms as T
import os

def load_image_cv2(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    return img

def clahe_enhance_bgr(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def pillow_to_tensor(image_pil, size):
    tf = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
    ])
    return tf(image_pil).unsqueeze(0)

def preprocess_for_model(img_path, model_name):
    """
    Returns a torch tensor (1,C,H,W) built from PIL for the requested model_name.
    Supported model_name keys: 'efficientnet_b0' (512), 'resnet50' (384), 'vit_small16' (224)
    """
    size_map = {
        'efficientnet_b0': 512,
        'resnet50': 384,
        'vit_small16': 224
    }
    size = size_map.get(model_name, 224)
    img = Image.open(img_path).convert("RGB")
    # center pad to square preserving retina circle
    w, h = img.size
    max_side = max(w, h)
    pad_w = (max_side - w) // 2
    pad_h = (max_side - h) // 2
    img = ImageOps.expand(img, border=(pad_w, pad_h, pad_w, pad_h), fill=(0,0,0))
    tensor = pillow_to_tensor(img, size)
    return tensor

def preprocess_for_model_cv2(img_path, model_name, apply_clahe=True):
    """
    cv2 path: returns (rgb_numpy, torch_tensor).
    rgb_numpy: HxWx3 in RGB (uint8)
    torch_tensor: 1xCxHxW normalized
    """
    bgr = load_image_cv2(img_path)
    if apply_clahe:
        try:
            bgr = clahe_enhance_bgr(bgr)
        except Exception:
            pass
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # create temporary PIL from rgb to use same transforms
    pil = Image.fromarray(rgb)
    # reuse preprocess_for_model (which pads and resizes)
    tensor = preprocess_for_model(img_path, model_name)
    return rgb, tensor

def prepare_ml_features(metadata_dict, scaler_feature_names):
    """
    Build numeric feature array aligned to scaler_feature_names (list-like).
    Strings like Yes/No or gender are encoded to simple numeric values.
    """
    import numpy as np
    feat = []
    for k in scaler_feature_names:
        # try different key naming styles
        val = None
        for candidate in (k, k.lower(), k.replace(' ', '_').lower()):
            if candidate in metadata_dict:
                val = metadata_dict[candidate]
                break
        if val is None:
            # fallback to metadata_dict.get(k,0)
            val = metadata_dict.get(k, 0)
        # convert
        try:
            fv = float(val)
        except Exception:
            if isinstance(val, str):
                v = val.strip().lower()
                if v in ('yes','y','true','1'):
                    fv = 1.0
                elif v in ('no','n','false','0'):
                    fv = 0.0
                elif v in ('male','m'):
                    fv = 1.0
                elif v in ('female','f'):
                    fv = 0.0
                else:
                    try:
                        fv = float(v)
                    except:
                        fv = 0.0
            else:
                fv = 0.0
        feat.append(fv)
    return np.array(feat).reshape(1, -1)
