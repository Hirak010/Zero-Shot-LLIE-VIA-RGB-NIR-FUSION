import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Imports from Colie project (assuming these files are in the same directory or PYTHONPATH)
from utils import get_image, get_coords, get_patches, interpolate_image, filter_up, get_v_component, replace_v_component
from loss import L_exp, L_TV
from siren import INF
from color import rgb2hsv_torch, hsv2rgb_torch

# Helper functions from enhancement.py (adapted for direct use)
# -----------------------------------------------------------------------------
# 1. FUSION HELPERS
# -----------------------------------------------------------------------------
def _get_max(img: np.ndarray, cx: int, cy: int, win: int) -> float:
    l = max(0, cy - win // 2)
    t = max(0, cx - win // 2)
    r = min(img.shape[1] - 1, cy + win // 2)
    b = min(img.shape[0] - 1, cx + win // 2)
    # Ensure slice is not empty before calling max
    if t > b or l > r:
        return 0.0 # Or handle appropriately, maybe return img[cx, cy]?
    return float(np.max(img[t : b + 1, l : r + 1]))

def _get_min(img: np.ndarray, cx: int, cy: int, win: int) -> float:
    l = max(0, cy - win // 2)
    t = max(0, cx - win // 2)
    r = min(img.shape[1] - 1, cy + win // 2)
    b = min(img.shape[0] - 1, cx + win // 2)
     # Ensure slice is not empty before calling min
    if t > b or l > r:
        return 0.0 # Or handle appropriately, maybe return img[cx, cy]?
    return float(np.min(img[t : b + 1, l : r + 1]))

def local_contrast(img: np.ndarray, alpha: float = 0.5, win: int = 5) -> np.ndarray:
    # Ensure input is float32 for Sobel
    img_float = img.astype(np.float32)
    dx = cv2.Sobel(img_float, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(img_float, cv2.CV_32F, 0, 1)
    amplitude = np.sqrt(dx ** 2 + dy ** 2)
    out = np.empty_like(img_float, np.float32)
    # Pad image to handle borders safely in get_max/min
    img_padded = cv2.copyMakeBorder(img_float, win//2, win//2, win//2, win//2, cv2.BORDER_REPLICATE)
    amp_padded = cv2.copyMakeBorder(amplitude, win//2, win//2, win//2, win//2, cv2.BORDER_REPLICATE)

    for x in range(img_float.shape[0]):
        for y in range(img_float.shape[1]):
            # Use padded coordinates
            px, py = x + win//2, y + win//2
            # Extract window from padded image
            img_win = img_padded[px - win//2 : px + win//2 + 1, py - win//2 : py + win//2 + 1]
            amp_win = amp_padded[px - win//2 : px + win//2 + 1, py - win//2 : py + win//2 + 1]

            if img_win.size == 0 or amp_win.size == 0: # Check if window extraction failed
                 max_i, min_i, max_a = 0.0, 0.0, 0.0 # Default values or handle error
            else:
                 max_i = float(np.max(img_win))
                 min_i = float(np.min(img_win))
                 max_a = float(np.max(amp_win))

            out[x, y] = alpha * (max_i - min_i) + (1.0 - alpha) * max_a
    return out


def fusion_map(lc_y: np.ndarray, lc_nir: np.ndarray, red: float = 0.7) -> np.ndarray:
    # Ensure inputs are float32
    lc_y_f = lc_y.astype(np.float32)
    lc_nir_f = lc_nir.astype(np.float32)
    # Calculate fusion map with epsilon for stability
    fmap = (np.maximum(0.0, (lc_nir_f - lc_y_f) * red) / (lc_nir_f + 1e-6))
    return fmap.astype(np.float32)


def high_pass(nir: np.ndarray, ksize: int = 19, strength: float = 0.7) -> np.ndarray:
    # Ensure nir is float32 for calculations
    nir_f = nir.astype(np.float32)
    base = cv2.GaussianBlur(nir_f, (ksize, ksize), 0)
    # Ensure base is also float32 before subtraction
    detail = (nir_f - base.astype(np.float32)) * strength
    return detail


def fuse_rgb_nir(rgb: np.ndarray, nir: np.ndarray, *, red: float = 0.7, hp_strength: float = 0.7) -> np.ndarray:
    # Ensure NIR is grayscale (single channel)
    if len(nir.shape) == 3 and nir.shape[2] == 3:
        nir_gray = cv2.cvtColor(nir, cv2.COLOR_BGR2GRAY)
    elif len(nir.shape) == 2:
        nir_gray = nir
    else:
        raise ValueError("NIR image must be grayscale or convertible to grayscale")

    # Ensure RGB is BGR format
    if rgb.shape[2] != 3:
         raise ValueError("RGB image must have 3 channels (BGR format expected by cvtColor)")

    # Convert RGB to YUV and extract Y channel (Luminance)
    try:
        yuv = cv2.cvtColor(rgb, cv2.COLOR_BGR2YUV)
        y = yuv[:, :, 0].astype(np.float32)
    except cv2.error as e:
        print(f"Error converting RGB to YUV: {e}")
        print(f"RGB image shape: {rgb.shape}, dtype: {rgb.dtype}")
        # As a fallback or debug step, maybe try converting RGB to Grayscale directly
        # y = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Or re-raise the exception if conversion is critical
        raise e


    # Calculate local contrast for Y and NIR
    lc_y = local_contrast(y)
    lc_nir = local_contrast(nir_gray.astype(np.float32)) # Ensure NIR is float for local_contrast

    # Calculate fusion map
    fmap = fusion_map(lc_y, lc_nir, red)

    # Calculate high-pass detail from NIR
    nir_detail = high_pass(nir_gray, strength=hp_strength) # Use grayscale NIR for high_pass

    # Add detail to RGB channels using the fusion map
    fused = rgb.astype(np.float32)
    for c in range(3):
        fused[:, :, c] += fmap * nir_detail # fmap and nir_detail should broadcast correctly

    # Clip values and convert back to uint8
    return np.clip(fused, 0, 255).astype(np.uint8)

# -----------------------------------------------------------------------------
# 2. GUIDED-FILTER DENOISING
# -----------------------------------------------------------------------------
def guided_denoise(rgb: np.ndarray, nir: np.ndarray, radius: int = 8, eps: float = 1e-3) -> np.ndarray:
    """Edge‑preserving denoise: NIR is guide, RGB is src."""
    # Ensure NIR is grayscale (single channel) and float32 [0, 1]
    if len(nir.shape) == 3 and nir.shape[2] == 3:
        nir_f = cv2.cvtColor(nir, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    elif len(nir.shape) == 2:
        nir_f = nir.astype(np.float32) / 255.0
    else:
        raise ValueError("NIR image must be grayscale or convertible to grayscale for guided filter")

    # Ensure RGB is float32 [0, 1]
    rgb_f = rgb.astype(np.float32) / 255.0

    # Check if ximgproc and guidedFilter are available
    use_guided_filter = hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter")

    if use_guided_filter:
        try:
            den = cv2.ximgproc.guidedFilter(guide=nir_f, src=rgb_f, radius=radius, eps=eps, dDepth=-1)
        except cv2.error as e:
             print(f"[WARN] cv2.ximgproc.guidedFilter failed: {e}. Falling back to bilateral filter.", file=sys.stderr)
             use_guided_filter = False # Force fallback
    
    if not use_guided_filter:
        # Fallback: bilateral filter (NIR guidance is lost here as cv2.bilateralFilter doesn't take a separate guide)
        print("[INFO] Using standard bilateral filter (NIR guidance not used in fallback)", file=sys.stderr)
        sig_sp, sig_col = float(radius) * 2, eps * 100  # Heuristics, adjust as needed
        # Apply bilateral filter channel-wise if needed, or directly on rgb_f
        den = cv2.bilateralFilter(rgb_f, d=-1, sigmaColor=sig_col, sigmaSpace=sig_sp)

    # Convert back to uint8 [0, 255]
    return (den * 255.0).clip(0, 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# 3. FILE HANDLING & MAIN PIPELINE
# -----------------------------------------------------------------------------
def get_matching_pairs(rgb_dir: Path, nir_dir: Path) -> List[Tuple[Path, Path]]:
    """Find matching RGB and NIR image pairs based on filename."""
    print(f"Scanning RGB directory: {rgb_dir}")
    rgb_files = {f.stem: f for f in rgb_dir.glob("*") if f.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}}
    print(f"Found {len(rgb_files)} potential RGB files.")

    print(f"Scanning NIR directory: {nir_dir}")
    nir_files = {f.stem: f for f in nir_dir.glob("*") if f.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}}
    print(f"Found {len(nir_files)} potential NIR files.")

    # Find common filenames (stems)
    common_names = set(rgb_files.keys()) & set(nir_files.keys())
    print(f"Found {len(common_names)} matching filenames.")

    if not common_names:
         print("Warning: No matching filenames found between RGB and NIR directories.")
         # Consider searching based on prefixes if names don't match exactly
         # e.g., if RGB is 'scene1_rgb.png' and NIR is 'scene1_nir.png'

    # Create pairs using the common stems
    pairs = []
    for name in sorted(list(common_names)): # Sort for consistent processing order
        pairs.append((rgb_files[name], nir_files[name]))

    return pairs


def tensor_to_cv2(tensor_img):
    """Converts a PyTorch tensor (C, H, W) [0,1] to OpenCV image (H, W, C) [0,255] BGR."""
    # Ensure tensor is on CPU and detached
    img_np = tensor_img.detach().cpu().numpy()
    # Remove batch dimension if present (e.g., shape [1, C, H, W])
    if img_np.ndim == 4 and img_np.shape[0] == 1:
        img_np = img_np.squeeze(0)
    # Transpose from (C, H, W) to (H, W, C)
    img_np = np.transpose(img_np, (1, 2, 0))
    # Scale from [0, 1] to [0, 255] and convert to uint8
    img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
    # Convert RGB to BGR for OpenCV
    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_cv2

def main():
    parser = argparse.ArgumentParser(description='Combined CoLIE Enhancement Pipeline')
    # Input/Output Args
    parser.add_argument('--rgb_dir', type=str, default='RGB/', help='Directory containing low-light RGB images')
    parser.add_argument('--nir_dir', type=str, default='NIR/', help='Directory containing corresponding NIR images')
    parser.add_argument('--out_dir', type=str, default='output_combined/', help='Output directory for final images')

    # Colie Args
    parser.add_argument('--down_size', type=int, default=256, help='Colie downsampling size')
    parser.add_argument('--epochs', type=int, default=100, help='Colie training epochs per image')
    parser.add_argument('--window', type=int, default=1, help='Colie context window size')
    parser.add_argument('--L', type=float, default=0.5, help='Colie exposure level target')
    parser.add_argument('--alpha', type=float, required=True, help='Colie loss_spa weight')
    parser.add_argument('--beta', type=float, required=True, help='Colie loss_tv weight')
    parser.add_argument('--gamma', type=float, required=True, help='Colie loss_exp weight')
    parser.add_argument('--delta', type=float, required=True, help='Colie loss_sparsity weight')

    # Enhancement Args
    parser.add_argument('--radius', type=int, default=2, help='Guided-filter radius (pixels)')
    parser.add_argument('--eps', type=float, default=1e-3, help='Guided-filter epsilon (regularisation)')
    parser.add_argument('--fusion_red', type=float, default=0.7, help='Fusion map reduction factor')
    parser.add_argument('--fusion_hp_strength', type=float, default=0.7, help='Fusion high-pass strength')


    opt = parser.parse_args()

    # --- Setup ---
    rgb_dir = Path(opt.rgb_dir)
    nir_dir = Path(opt.nir_dir)
    out_dir = Path(opt.out_dir)

    if not rgb_dir.is_dir():
        raise SystemExit(f"RGB directory not found: {rgb_dir}")
    if not nir_dir.is_dir():
        raise SystemExit(f"NIR directory not found: {nir_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Find Image Pairs ---
    image_pairs = get_matching_pairs(rgb_dir, nir_dir)
    if not image_pairs:
        raise SystemExit("No matching RGB-NIR image pairs found. Ensure filenames (without extension) match.")

    print(f"Found {len(image_pairs)} image pairs. Starting processing...")

    # --- Processing Loop ---
    for rgb_path, nir_path in tqdm(image_pairs, desc="Processing Images"):
        try:
            # --- 1. Load Images ---
            # Load RGB for Colie (using its loader for tensor conversion)
            img_rgb_tensor = get_image(str(rgb_path)).to(device) # Shape: [1, C, H, W]
            # Load NIR for Enhancement (using OpenCV)
            img_nir_cv = cv2.imread(str(nir_path), cv2.IMREAD_GRAYSCALE)
            if img_nir_cv is None:
                print(f"Warning: Failed to load NIR image {nir_path}. Skipping pair.")
                continue

            # --- 2. Colie Enhancement ---
            img_hsv = rgb2hsv_torch(img_rgb_tensor)
            img_v = get_v_component(img_hsv) # Shape: [1, 1, H, W]
            
            # Determine original size for upsampling later
            original_height, original_width = img_v.shape[2], img_v.shape[3]

            # Downsample V channel
            img_v_lr = interpolate_image(img_v, opt.down_size, opt.down_size) # Shape: [1, 1, down_size, down_size]
            coords = get_coords(opt.down_size, opt.down_size).to(device) # Shape: [1, down_size*down_size, 2]
            patches = get_patches(img_v_lr, opt.window).to(device) # Shape: [1, down_size*down_size, window*window]

            # Initialize SIREN model
            img_siren = INF(patch_dim=opt.window**2, num_layers=4, hidden_dim=256, add_layer=2).to(device)
            optimizer = torch.optim.Adam(img_siren.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=3e-4)

            # Loss functions
            l_exp = L_exp(16, opt.L).to(device)
            l_TV = L_TV().to(device)

            # Colie training loop per image
            for epoch in range(opt.epochs):
                img_siren.train()
                optimizer.zero_grad()

                illu_res_lr = img_siren(patches, coords) # Shape: [1, down_size*down_size, 1]
                illu_res_lr = illu_res_lr.view(1, 1, opt.down_size, opt.down_size) # Reshape
                illu_lr = illu_res_lr + img_v_lr # Predicted illumination map

                # Avoid division by zero
                img_v_fixed_lr = img_v_lr / (illu_lr + 1e-4)

                # Calculate losses
                loss_spa = torch.mean(torch.abs(torch.pow(illu_lr - img_v_lr, 2)))
                loss_tv = l_TV(illu_lr)
                loss_exp = torch.mean(l_exp(illu_lr))
                # Sparsity loss might need adjustment depending on expected range of img_v_fixed_lr
                loss_sparsity = torch.mean(torch.pow(img_v_fixed_lr, 2)) # Original used mean(abs(I)), let's try mean square

                loss = loss_spa * opt.alpha + loss_tv * opt.beta + loss_exp * opt.gamma + loss_sparsity * opt.delta
                loss.backward()
                optimizer.step()
                # Optional: Add logging for loss per epoch/image

            # Upsample the enhanced V channel using guided filter
            # The filter uses the high-resolution img_v as the guide (x_hr)
            img_v_fixed = filter_up(img_v_lr, img_v_fixed_lr, img_v) # Removed H and W args

            # Reconstruct HSV and then RGB
            img_hsv_fixed = replace_v_component(img_hsv, img_v_fixed)
            img_rgb_fixed_tensor = hsv2rgb_torch(img_hsv_fixed)

            # Normalize the result (important step from original colie.py)
            img_rgb_fixed_tensor = img_rgb_fixed_tensor / torch.max(img_rgb_fixed_tensor) # Normalize to [0, 1]

            # --- 3. Convert Colie Output for OpenCV ---
            img_rgb_colie_cv = tensor_to_cv2(img_rgb_fixed_tensor) # HWC, BGR, uint8 [0, 255]

            # --- 4. Enhancement Fusion & Denoising ---
            # Fuse Colie's output with NIR
            img_fused_cv = fuse_rgb_nir(img_rgb_colie_cv, img_nir_cv, red=opt.fusion_red, hp_strength=opt.fusion_hp_strength)

            # Denoise the fused image using NIR as guide
            img_denoised_cv = guided_denoise(img_fused_cv, img_nir_cv, radius=opt.radius, eps=opt.eps)

            # --- 5. Save Result ---
            output_filename = out_dir / rgb_path.name # Use original RGB filename for output
            cv2.imwrite(str(output_filename), img_denoised_cv)

        except Exception as e:
            print(f"\nError processing pair ({rgb_path.name}, {nir_path.name}): {e}")
            import traceback
            traceback.print_exc()
            # Continue to the next pair

    print(f"\nProcessing complete. Results saved in: {out_dir}")

if __name__ == "__main__":
    main()
