# Zero-Shot-LLIE-VIA-RGB-NIR-FUSION
A Zero-Shot approach to low light image enhancement using RGB-NIR Implicit Fusion

# Zero-Shot RGB + NIR Low-Light Enhancement  
### Code for *“Zero-Shot Low-Light Image Enhancement via RGB–NIR Implicit Fusion”* (M.Sc. thesis, IIT ISM Dhanbad)

`combined_pipeline.py` is a one-stop script that reproduces the full pipeline proposed in the thesis:

1. **CoLIE illumination optimisation** – fits a per-image SIREN-based implicit neural function to the HSV-Value channel.  
2. **Adaptive RGB ↔ NIR fusion** – injects high-frequency NIR detail with a pixel-wise contrast-based weighting map.  
3. **Edge-preserving denoising** – guided filter (or bilateral fall-back) that uses the NIR frame as guidance.  
4. **Export** – enhanced RGB images are written to `--out_dir`.

The whole process is *zero-shot*: no pre-training, no paired GT, no synthetic noise.

---

## Repository layout


