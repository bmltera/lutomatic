"""
seg_reinhard_coloring.py

Performs segmentation-aware color transfer between an input and target image using the Reinhard method.
- Uses SAM (Segment Anything Model) for image segmentation.
- Uses CLIP for segment labeling.
- Computes color statistics in LAB space for both global and segmentation-aware regions.
- Blends global and segmentation-aware statistics based on a configurable weight.
- Applies color transfer and writes out overlays, transferred images, and a LUT (.cube) for use in color grading.
"""

import os
import cv2
import json
import numpy as np
from datetime import datetime
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import torch
import clip
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator

# CONFIGURATION
INPUT_IMG  = "images/input3.png"      # Path to input image
TARGET_IMG = "images/target4.png"     # Path to target image
MODEL_TYPE = "vit_b"                  # SAM model type
MODEL_PATH = "sam_vit_b_01ec64.pth"   # Path to SAM model weights
WITH_GPU   = torch.cuda.is_available()
DEVICE     = "cuda" if WITH_GPU else "cpu"

# PARAMETERS
SEGMENTATION_WEIGHT = 40  # 0 = global only, 100 = segmentation only, 50 = blend
EXPOSURE_MATCH = False    # Toggle exposure matching
SHOW_SEGMENTATION = True  # Toggle segmentation overlay output

# If True, apply the generated LUT to the input image and save as input_LUT_applied.png
APPLY_LUT = True

print("test")

# Load label names for CLIP classification
with open("labels.json", "r") as f:
    LABELS = json.load(f)

def make_output_dir():
    """
    Creates a timestamped output directory for results.
    Returns the path to the created directory.
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = os.path.join("output", ts)
    os.makedirs(out, exist_ok=True)
    return out

def load_models():
    """
    Loads the SAM segmentation model, CLIP model, and prepares text embeddings for labels.
    Returns:
        predictor: SAM predictor object
        mask_gen: SAM automatic mask generator
        clip_model: CLIP model
        clip_pre: CLIP image preprocessor
        txt_emb: Precomputed CLIP text embeddings for labels
    """
    sam = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_PATH).to(DEVICE)
    predictor = SamPredictor(sam)
    mask_gen  = SamAutomaticMaskGenerator(sam)
    clip_model, clip_pre = clip.load("ViT-B/32", device=DEVICE)
    # Precompute text embeddings for all labels
    with torch.no_grad():
        tokens  = clip.tokenize(LABELS).to(DEVICE)
        txt_emb = clip_model.encode_text(tokens)
        txt_emb /= txt_emb.norm(dim=-1, keepdim=True)
    return predictor, mask_gen, clip_model, clip_pre, txt_emb

def resize_max(img, max_dim=512):
    """
    Resizes an image so its largest dimension is max_dim, preserving aspect ratio.
    """
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1.0:
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img

def segment_and_label(img_bgr, predictor, mask_gen, clip_model, clip_pre, txt_emb):
    """
    Segments the image using SAM and labels each segment using CLIP.
    Returns a list of dicts: { "mask": mask, "label": label }
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    masks = mask_gen.generate(rgb)
    results = []
    for m in masks:
        seg = m["segmentation"]
        ys, xs = np.where(seg)
        if len(xs) == 0:
            continue
        # Crop the segment and preprocess for CLIP
        crop = rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        pil  = clip_pre(Image.fromarray(crop)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = clip_model.encode_image(pil)
            emb /= emb.norm(dim=-1, keepdim=True)
            sims = (txt_emb @ emb.T).squeeze(1)
            idx  = sims.argmax().item()
        results.append({"mask": seg, "label": LABELS[idx]})
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
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = None
    for idx, seg in enumerate(segments):
        ys, xs = np.where(seg["mask"])
        if len(xs) == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())
        text = seg['label']
        draw.text((cx, cy), text, fill=(255, 255, 255), font=font, anchor="mm")
    arr = np.array(pil_overlay).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def collect_lab_stats(segments, img_bgr, classes):
    """
    Computes mean and std of LAB color values for the specified classes in the given segments.
    Returns (mean, std) as np arrays.
    """
    lab_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    pix = defaultdict(list)
    for seg in segments:
        lbl = seg["label"]
        if lbl in classes:
            mask = seg["mask"]
            pix[lbl].append(lab_img[mask])
    # Concatenate all pixels for the selected classes
    all_pixels = np.concatenate([np.vstack(pix[l]) for l in classes], axis=0)
    mean = all_pixels.mean(axis=0)
    std  = all_pixels.std(axis=0) + 1e-6
    return mean, std

def collect_lab_stats_global(img_bgr):
    """
    Computes global mean and std of LAB color values for the entire image.
    Returns (mean, std) as np arrays.
    """
    lab_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    mean = lab_img.reshape(-1, 3).mean(axis=0)
    std  = lab_img.reshape(-1, 3).std(axis=0) + 1e-6
    return mean, std

def match_exposure(input_bgr, meanL_src, meanL_tgt):
    """
    Adjusts the L (lightness) channel of the input image to match the target mean L.
    Returns the exposure-matched image (BGR).
    """
    lab = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = cv2.split(lab)
    scale = meanL_tgt / (meanL_src + 1e-6)
    L = np.clip(L * scale, 0, 255)
    lab_matched = cv2.merge([L, a, b]).astype(np.uint8)
    return cv2.cvtColor(lab_matched, cv2.COLOR_LAB2BGR)

def reinhard_transfer(input_bgr, mean1, std1, mean2, std2):
    """
    Applies Reinhard color transfer from source (mean1, std1) to target (mean2, std2) in LAB space.
    Returns the color-transferred image (BGR).
    """
    lab = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    # Per-channel transformation
    lab = (lab - mean1) * (std2 / std1) + mean2
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return out

def srgb_to_linear(rgb):
    """
    Convert sRGB [0,1] to linear RGB [0,1].
    """
    rgb = np.clip(rgb, 0, 1)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return linear

def linear_to_srgb(rgb):
    """
    Convert linear RGB [0,1] to sRGB [0,1].
    """
    rgb = np.clip(rgb, 0, 1)
    srgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * (rgb ** (1/2.4)) - 0.055)
    return srgb

def write_image_based_lut(path, input_img, output_img, size=33, k=8):
    """
    Generate a 3D LUT that, when applied to input_img, produces output_img.
    For each LUT grid point (input color), use k-nearest neighbors in input_img and interpolate the corresponding output_img color.
    Both images must be the same size and in uint8 BGR format.
    LUT is written in .cube format, sRGB gamma-encoded [0,1].
    Uses KD-tree for fast nearest neighbor search and inverse distance weighting.
    Prints first few LUT entries for diagnostics.

    NOTE: The mapping is FROM input_img TO output_img, so the LUT can be applied to input_img to produce output_img.
    """
    from scipy.spatial import cKDTree

    assert input_img.shape == output_img.shape, "Input and output images must have the same shape"
    h, w, _ = input_img.shape
    # Convert to RGB for LUT mapping (OpenCV loads as BGR)
    input_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).reshape(-1, 3).astype(np.float32) / 255.0
    output_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB).reshape(-1, 3).astype(np.float32) / 255.0

    # Build KD-tree for fast nearest neighbor search
    tree = cKDTree(input_rgb)

    with open(path, "w") as f:
        f.write(f'TITLE "ImageBasedLUT ({size}x{size}x{size})"\n')
        f.write(f"LUT_3D_SIZE {size}\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n")

        # Build LUT grid
        grid = np.linspace(0, 1, size)
        diagnostic_prints = 0
        for r in range(size):
            for g in range(size):
                for b in range(size):
                    rgb = np.array([grid[r], grid[g], grid[b]])
                    # k-nearest neighbors and inverse distance weighting
                    dists, idxs = tree.query(rgb, k=k)
                    dists = np.maximum(dists, 1e-6)
                    weights = 1.0 / dists
                    weights /= weights.sum()
                    out_rgb = np.sum(output_rgb[idxs] * weights[:, None], axis=0)
                    f.write(f"{out_rgb[2]:.6f} {out_rgb[1]:.6f} {out_rgb[0]:.6f}\n")

                    # Diagnostic: print first few LUT entries
                    if diagnostic_prints < 10:
                        print(f"LUT grid RGB: {rgb}, mapped to output RGB: {out_rgb}, input nearest: {input_rgb[idxs[0]]}, output nearest: {output_rgb[idxs[0]]}")
                        diagnostic_prints += 1

def blend_stats(seg_stat, global_stat, weight):
    """
    Blends segmentation-aware and global statistics according to the given weight.
    weight: 0..100 (0=global only, 100=segmentation only)
    Returns (mean, std) as np arrays.
    """
    alpha = weight / 100.0
    mean = alpha * seg_stat[0] + (1 - alpha) * global_stat[0]
    std  = alpha * seg_stat[1] + (1 - alpha) * global_stat[1]
    return mean, std

def load_cube_lut(cube_path):
    """
    Loads a 3D LUT from a .cube file.
    Returns: lut (size, size, size, 3), size (int)
    """
    lut = []
    size = None
    with open(cube_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("TITLE"):
                continue
            if line.startswith("LUT_3D_SIZE"):
                size = int(line.split()[1])
                continue
            if line[0].isdigit() or line[0] == "-":
                vals = [float(x) for x in line.split()]
                lut.append(vals)
    if size is None:
        raise ValueError("LUT_3D_SIZE not found in .cube file")
    lut = np.array(lut).reshape((size, size, size, 3))
    return lut, size

def apply_ocio_lut(img_bgr, lut_path):
    """
    Applies a .cube LUT to an image using OpenColorIO.
    Expects img_bgr as uint8, LUT as .cube file path.
    Returns: img_bgr_out (uint8)
    """
    import PyOpenColorIO as ocio

    # Convert BGR to RGB and scale to [0,1]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w, c = img_rgb.shape
    img_flat = img_rgb.reshape(-1, 3)

    # Set up OCIO config and processor for the LUT
    config = ocio.Config.CreateRaw()
    processor = config.getProcessor(ocio.FileTransform(src=lut_path, direction=ocio.TRANSFORM_DIR_FORWARD))
    cpu = processor.getDefaultCPUProcessor()

    # Apply LUT to all pixels (use only cpu.applyRGBs)
    img_lut = img_flat.copy()
    cpu.applyRGB(img_lut)
    img_lut = np.clip(np.array(img_lut), 0, 1).reshape(h, w, 3)
    img_lut_uint8 = (img_lut * 255).round().astype(np.uint8)
    img_bgr_out = cv2.cvtColor(img_lut_uint8, cv2.COLOR_RGB2BGR)
    return img_bgr_out

def main():
    """
    Main workflow:
    - Loads models and images
    - Segments and labels input/target images
    - Computes color statistics (global and segmentation-aware)
    - Blends statistics and applies Reinhard color transfer
    - Writes overlays, transferred images, and LUT to output directory
    """
    print("test main")

    print("Creating output directory...")
    out = make_output_dir()
    print("Loading models...")
    predictor, mask_gen, clip_model, clip_pre, txt_emb = load_models()
    print("Reading and resizing images...")
    inp = cv2.imread(INPUT_IMG)
    tgt = cv2.imread(TARGET_IMG)
    if inp is None or tgt is None:
        print("ERROR: Could not read input or target image.")
        return
    inp_s = resize_max(inp)
    tgt_s = resize_max(tgt)

    print("Segmenting and labeling input image...")
    seg_i = segment_and_label(inp_s, predictor, mask_gen, clip_model, clip_pre, txt_emb)
    print("Segmenting and labeling target image...")
    seg_t = segment_and_label(tgt_s, predictor, mask_gen, clip_model, clip_pre, txt_emb)
    c1 = set(s["label"] for s in seg_i)
    c2 = set(s["label"] for s in seg_t)
    common = sorted(c1 & c2)
    print(f"Found {len(common)} overlapping classes: {common}")
    if not common:
        print("No overlap")
        return

    # Show segmentation overlays if requested
    if SHOW_SEGMENTATION:
        print("Rendering segmentation overlays...")
        input_overlay = render_overlay(inp_s, seg_i)
        target_overlay = render_overlay(tgt_s, seg_t)
        cv2.imwrite(os.path.join(out, "input_overlay.png"), input_overlay)
        cv2.imwrite(os.path.join(out, "target_overlay.png"), target_overlay)

    # Compute LAB statistics for shared classes (segmentation-aware)
    print("Computing LAB stats on shared classes…")
    seg_m1, seg_s1 = collect_lab_stats(seg_i, inp_s, common)
    seg_m2, seg_s2 = collect_lab_stats(seg_t, tgt_s, common)
    print(f"Segmentation-aware input mean/std: {seg_m1} / {seg_s1}")
    print(f"Segmentation-aware target mean/std: {seg_m2} / {seg_s2}")

    # Compute global LAB statistics
    print("Computing global LAB stats…")
    glob_m1, glob_s1 = collect_lab_stats_global(inp_s)
    glob_m2, glob_s2 = collect_lab_stats_global(tgt_s)
    print(f"Global input mean/std: {glob_m1} / {glob_s1}")
    print(f"Global target mean/std: {glob_m2} / {glob_s2}")

    # Blend statistics according to SEGMENTATION_WEIGHT
    blend_m1, blend_s1 = blend_stats((seg_m1, seg_s1), (glob_m1, glob_s1), SEGMENTATION_WEIGHT)
    blend_m2, blend_s2 = blend_stats((seg_m2, seg_s2), (glob_m2, glob_s2), SEGMENTATION_WEIGHT)
    print(f"Blended input mean/std: {blend_m1} / {blend_s1}")
    print(f"Blended target mean/std: {blend_m2} / {blend_s2}")

    # Helper for exposure scale calculation
    def get_exposure_scale(meanL_src, meanL_tgt):
        return meanL_tgt / (meanL_src + 1e-6)

    # Apply segmentation-aware color transfer
    print("Applying segmentation-aware Reinhard color transfer…")
    inp_seg = inp
    exposure_scale_seg = 1.0
    if EXPOSURE_MATCH:
        exposure_scale_seg = get_exposure_scale(seg_m1[0], seg_m2[0])
        print(f"Segmentation-aware exposure scale: {exposure_scale_seg}")
        inp_seg = match_exposure(inp, seg_m1[0], seg_m2[0])
    out_img_seg = reinhard_transfer(inp_seg, seg_m1, seg_s1, seg_m2, seg_s2)
    cv2.imwrite(os.path.join(out, "input_reinhard_classaware.png"), out_img_seg)

    # Apply global color transfer
    print("Applying global Reinhard color transfer…")
    inp_glob = inp
    exposure_scale_glob = 1.0
    if EXPOSURE_MATCH:
        exposure_scale_glob = get_exposure_scale(glob_m1[0], glob_m2[0])
        print(f"Global exposure scale: {exposure_scale_glob}")
        inp_glob = match_exposure(inp, glob_m1[0], glob_m2[0])
    out_img_glob = reinhard_transfer(inp_glob, glob_m1, glob_s1, glob_m2, glob_s2)
    cv2.imwrite(os.path.join(out, "input_reinhard_global.png"), out_img_glob)

    # Apply blended color transfer if blending is enabled
    if SEGMENTATION_WEIGHT not in (0, 100):
        print("Applying blended Reinhard color transfer…")
        inp_blend = inp
        exposure_scale_blend = 1.0
        if EXPOSURE_MATCH:
            exposure_scale_blend = get_exposure_scale(blend_m1[0], blend_m2[0])
            print(f"Blended exposure scale: {exposure_scale_blend}")
            inp_blend = match_exposure(inp, blend_m1[0], blend_m2[0])
        out_img_blend = reinhard_transfer(inp_blend, blend_m1, blend_s1, blend_m2, blend_s2)
        cv2.imwrite(os.path.join(out, "input_reinhard_blend.png"), out_img_blend)

    # Save original input and target images for comparison
    cv2.imwrite(os.path.join(out, "input_orig.png"), inp)
    cv2.imwrite(os.path.join(out, "input_reinhard_classaware.png"), out_img_seg)
    cv2.imwrite(os.path.join(out, "target_orig.png"), tgt)

    # Write LUT (.cube) for blended transfer
    lut_path = os.path.join(out, "reinhard_blend.cube")
    print("Writing image-based LUT from input_orig.png to input_reinhard_classaware.png...")
    # Load the two images
    input_orig_path = os.path.join(out, "input_orig.png")
    input_reinhard_path = os.path.join(out, "input_reinhard_classaware.png")
    input_img = cv2.imread(input_orig_path)
    output_img = cv2.imread(input_reinhard_path)
    if input_img is None or output_img is None:
        print("ERROR: Could not read input_orig.png or input_reinhard_classaware.png for LUT generation.")
        return
    write_image_based_lut(lut_path, input_img, output_img, size=33)

    # Optionally apply the generated LUT to the input image and save
    if APPLY_LUT:
        print("Applying generated LUT to input image using OpenColorIO...")
        out_lut_path = os.path.join(out, "output_LUT.png")
        try:
            img_lut_applied = apply_ocio_lut(inp, lut_path)
            cv2.imwrite(out_lut_path, img_lut_applied)
            print(f"LUT-applied result saved as {out_lut_path}")
        except ImportError:
            print("PyOpenColorIO is not installed. Please install it with 'pip install opencolorio' to use OCIO LUT application.")

    print(f"Done. Results in {out}")

if __name__ == "__main__":
    main()
