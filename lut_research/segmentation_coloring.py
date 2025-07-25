#!/usr/bin/env python
"""
AutoLUT – baseline segmentation demo
------------------------------------
1. Segment both input and target images with Segment Anything Model (SAM).
2. Label each mask using CLIP zero‑shot classification.
3. Save overlays + class lists into ./output/<run_timestamp>/.

Install deps first:
    pip install segment-anything clip transformers torch torchvision pillow opencv-python numpy
Place a SAM checkpoint (e.g. sam_vit_b_01ec64.pth) in the same folder or edit MODEL_PATH.

Author: <your‑name>
"""

import os, cv2, time, json
from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from median_color import median_color

import torch
import clip
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator

# -------- CONFIG ----------------------------------------------------------------

INPUT_IMG  = "images/input.png"   # Path to input image
TARGET_IMG = "images/target.png"  # Path to target image
MODEL_TYPE = "vit_b"              # SAM model type
MODEL_PATH = "sam_vit_b_01ec64.pth"   # download from Meta’s SAM repo

# Load label names for CLIP classification
with open("labels.json", "r") as f:
    LABELS = json.load(f)

# Optional: true‑type font for drawing labels
FONT_PATH  = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# -----------------------------------------------------------------------------


def make_output_dir():
    """
    Creates a timestamped output directory for results.
    Returns the path to the created directory.
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join("output", ts)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def load_models(device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Loads the SAM segmentation model, CLIP model, and prepares text embeddings for labels.
    Returns:
        predictor: SAM predictor object
        mask_generator: SAM automatic mask generator
        clip_model: CLIP model
        clip_preprocess: CLIP image preprocessor
        text_emb: Precomputed CLIP text embeddings for labels
        device: torch device string
    """
    # SAM
    sam = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_PATH).to(device)
    predictor = SamPredictor(sam)
    mask_generator = SamAutomaticMaskGenerator(sam)

    # CLIP
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    with torch.no_grad():
        tokens = clip.tokenize(LABELS).to(device)
        text_emb = clip_model.encode_text(tokens)
        text_emb /= text_emb.norm(dim=-1, keepdim=True)

    return predictor, mask_generator, clip_model, clip_preprocess, text_emb, device

def segment_and_label(img_bgr, mask_generator, clip_model, clip_pre, text_emb, device):
    """
    Segments the image using SAM and labels each segment using CLIP.
    Returns a list of dicts: { 'mask': mask, 'label': label, 'score': float }
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(img_rgb)  # List of dicts with 'segmentation' key
    results = []

    for mask_dict in masks:
        m = mask_dict["segmentation"]
        ys, xs = np.where(m)
        if len(xs) == 0:        # empty mask (rare)
            continue
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
        crop = img_rgb[y0:y1+1, x0:x1+1]
        pil = clip_pre(Image.fromarray(crop)).unsqueeze(0).to(device)

        with torch.no_grad():
            img_emb = clip_model.encode_image(pil)
            img_emb /= img_emb.norm(dim=-1, keepdim=True)
            sims = (text_emb @ img_emb.T).squeeze(1)     # (len(LABELS),)
            best = sims.argmax().item()
            label = LABELS[best]
            score = sims[best].item()

        results.append({"mask": m, "label": label, "score": score})
    return results

def render_overlay(img_bgr, segments, alpha=0.5):
    """
    Renders a colored overlay for each segment and draws the label at the segment centroid.
    Returns the overlay image (BGR).
    """
    overlay = img_bgr.copy()
    H, W = img_bgr.shape[:2]
    rng = np.random.default_rng(42)
    colors = rng.integers(0, 255, size=(len(segments), 3))

    for idx, seg in enumerate(segments):
        color = colors[idx].tolist()
        mask = seg["mask"]
        for c in range(3):
            overlay[..., c][mask] = (overlay[..., c][mask] * (1 - alpha) + alpha * color[c]).astype(np.uint8)

    # Draw labels at centroid
    pil_overlay = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_overlay)
    font = ImageFont.truetype(FONT_PATH, 20) if os.path.exists(FONT_PATH) else None
    for idx, seg in enumerate(segments):
        ys, xs = np.where(seg["mask"])
        cx, cy = int(xs.mean()), int(ys.mean())
        text = seg['label']
        draw.text((cx, cy), text, fill=(255,255,255), font=font, anchor="mm")
    arr = np.array(pil_overlay).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def save_outputs(out_dir, name, img_bgr, segments):
    """
    Saves the original image and overlay to the output directory.
    Returns the sorted list of unique class labels found in the segments.
    """
    cv2.imwrite(os.path.join(out_dir, f"{name}_orig.png"), img_bgr)
    cv2.imwrite(os.path.join(out_dir, f"{name}_overlay.png"), render_overlay(img_bgr, segments))

    classes = sorted(set(s["label"] for s in segments))
    return classes

def main():
    """
    Main workflow:
    - Loads models and images
    - Segments and labels input/target images
    - Saves overlays and class lists
    - Aggregates and compares median colors per class
    - Solves for a color correction matrix and LUT if classes overlap
    """
    out_dir = make_output_dir()
    predictor, mask_generator, clip_model, clip_pre, text_emb, device = load_models()

    # ---------- INPUT image ----------
    img_in  = cv2.imread(INPUT_IMG)
    seg_in  = segment_and_label(img_in, mask_generator, clip_model, clip_pre, text_emb, device)
    classes_in = save_outputs(out_dir, "input", img_in, seg_in)

    # ---------- TARGET image ----------
    img_tgt = cv2.imread(TARGET_IMG)
    seg_tgt = segment_and_label(img_tgt, mask_generator, clip_model, clip_pre, text_emb, device)
    classes_tgt = save_outputs(out_dir, "target", img_tgt, seg_tgt)

    # ---------- Write class list ----------
    txt_path = os.path.join(out_dir, "classes.txt")
    with open(txt_path, "w") as f:
        f.write("Input image classes:\n")
        for c in classes_in:
            f.write(f"  - {c}\n")
        f.write("\nTarget image classes:\n")
        for c in classes_tgt:
            f.write(f"  - {c}\n")

    # ---------- Write median colors (aggregated per class) ----------
    from collections import defaultdict

    def aggregate_median_colors(segments, img_bgr):
        """
        For each class, computes the median RGB color of all pixels in all segments of that class.
        Returns a dict: { label: (R, G, B) }
        """
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        color_dict = defaultdict(list)
        for seg in segments:
            color = median_color(img_rgb, seg["mask"])
            color_tuple = tuple(int(x) for x in color)
            color_dict[seg["label"]].append(color_tuple)
        # For each class, compute the median of all median colors
        final_medians = {}
        for label, colors in color_dict.items():
            arr = np.array(colors)
            median = tuple(int(x) for x in np.median(arr, axis=0))
            final_medians[label] = median
        return final_medians

    input_medians = aggregate_median_colors(seg_in, img_in)
    target_medians = aggregate_median_colors(seg_tgt, img_tgt)

    median_path = os.path.join(out_dir, "median_colors.txt")
    with open(median_path, "w") as f:
        f.write("Input image median colors:\n")
        for label, color in sorted(input_medians.items()):
            f.write(f"{label} {color}\n")
        f.write("\nTarget image median colors:\n")
        for label, color in sorted(target_medians.items()):
            f.write(f"{label} {color}\n")

        # Table for classes in both images
        common_classes = sorted(set(input_medians.keys()) & set(target_medians.keys()))
        if common_classes:
            f.write("\n\nInput image         Target image\n\n")
            for label in common_classes:
                color_in = input_medians[label]
                color_tgt = target_medians[label]
                # Euclidean distance in RGB
                diff = np.linalg.norm(np.array(color_in) - np.array(color_tgt))
                pct = 100 * diff / (np.sqrt(3) * 255)  # max possible distance
                f.write(f"{label:20s} median: {color_in}    {label:20s} median: {color_tgt}  % difference: {pct:.1f}%\n")

    # ---------- Generate matrix-based LUT and apply to input image ----------
    def solve_color_matrix(input_colors, target_colors):
        """
        Solve for a 3x4 affine color correction matrix that maps input_colors to target_colors.
        Returns a (3,4) matrix M such that [R',G',B'] = M @ [R,G,B,1]
        """
        X = np.hstack([np.array(input_colors), np.ones((len(input_colors), 1))])  # (N,4)
        Y = np.array(target_colors)  # (N,3)
        M, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)  # (4,3)
        return M.T  # (3,4)

    def apply_color_matrix(img_bgr, M):
        """
        Applies the affine color correction matrix M to the input image.
        Returns the color-corrected image (BGR).
        """
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        h, w, _ = img_rgb.shape
        flat = img_rgb.reshape(-1, 3)
        flat_aug = np.hstack([flat, np.ones((flat.shape[0], 1))])
        mapped = (M @ flat_aug.T).T
        mapped = np.clip(mapped, 0, 255).astype(np.uint8)
        img_rgb_out = mapped.reshape(h, w, 3)
        return cv2.cvtColor(img_rgb_out, cv2.COLOR_RGB2BGR)

    def write_matrix_cube_lut(lut_path, M, lut_size=33):
        """
        Writes a 3D LUT (.cube) file for the affine color correction matrix.
        """
        with open(lut_path, "w") as f:
            f.write("TITLE \"MatrixLUT\"\n")
            f.write(f"LUT_3D_SIZE {lut_size}\n")
            f.write("DOMAIN_MIN 0.0 0.0 0.0\n")
            f.write("DOMAIN_MAX 1.0 1.0 1.0\n")
            for r in range(lut_size):
                for g in range(lut_size):
                    for b in range(lut_size):
                        rgb = np.array([r, g, b]) / (lut_size - 1) * 255
                        rgb_aug = np.append(rgb, 1.0)
                        mapped = M @ rgb_aug
                        mapped = np.clip(mapped / 255.0, 0, 1)
                        f.write(f"{mapped[0]:.6f} {mapped[1]:.6f} {mapped[2]:.6f}\n")

    if common_classes:
        # If there are classes in both images, solve for a color correction matrix
        input_colors = [input_medians[label] for label in common_classes]
        target_colors = [target_medians[label] for label in common_classes]
        M = solve_color_matrix(input_colors, target_colors)
        lut_path = os.path.join(out_dir, "input_to_target_matrix.cube")
        write_matrix_cube_lut(lut_path, M, lut_size=33)

        # Apply matrix to input image and save result
        img_lut_applied = apply_color_matrix(img_in, M)
        cv2.imwrite(os.path.join(out_dir, "input_LUT_applied.png"), img_lut_applied)

    print(f"Done! Results in {out_dir}")

if __name__ == "__main__":
    main()
