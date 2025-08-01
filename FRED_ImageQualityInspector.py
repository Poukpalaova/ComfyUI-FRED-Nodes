# import torch
# import numpy as np
# import cv2
# from PIL import Image
# from imquality import brisque
# from skimage.color import rgb2gray
# import os

# HELP_MESSAGE = """\
# 📊 Image Quality Analysis:

# • BRISQUE Score (0–100+):
  # - Measures perceptual image quality without reference.
  # - Lower is better: 
      # < 30 → High quality
      # 30–60 → Medium
      # > 60 → Low
      # -1 → Invalid result

# • Blur Value (0–500+):
  # - Based on variance of Laplacian.
  # - Higher is sharper:
      # > 300 → Very sharp
      # 100–300 → Moderately sharp
      # < 100 → Likely blurry

# • SNR (Signal-to-Noise Ratio, in dB):
  # - Measures signal clarity vs noise.
  # - Higher is better:
      # > 25 dB → Low noise
      # 15–25 dB → Moderate noise
      # < 15 dB → Noisy

# • Compression Ratio (Raw/JPEG):
  # - Estimates compression level.
    # ~1.0 → Normal
    # < 0.7 → Likely lossy compression
    # > 1.2 → High quality or uncompressed

# Use a mask input to restrict blur and noise analysis to a specific region.
# BRISQUE is always calculated on the full image.
# """

# class FRED_ImageQualityInspector:
    # @classmethod
    # def INPUT_TYPES(cls):
        # return {
            # "required": {
                # "image": ("IMAGE",),
            # },
            # "optional": {
                # "mask": ("MASK",),
            # }
        # }

    # RETURN_TYPES = ("IMAGE", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "STRING")
    # RETURN_NAMES = ("GRAY_IMAGE", "BRISQUE_SCORE", "BLUR_VALUE", "SNR_VALUE", "COMPRESSION_RATIO", "help")
    # FUNCTION = "analyze_image"
    # CATEGORY = "👑FRED/analysis"

    # def analyze_image(self, image, mask=None):
        # image_tensor = image[0]  # (C, H, W)
        # image_np = image_tensor.cpu().numpy()
        # # Corrige automatiquement si nécessaire
        # if image_np.shape[0] == 3:
            # image_np = image_np.transpose(1, 2, 0)  # (C, H, W) → (H, W, C)
        # elif image_np.shape[2] == 3:
            # pass  # déjà bon (H, W, C)
        # else:
            # raise ValueError(f"Unexpected image shape for RGB: {image_np.shape}")
        # image_np = (image_np * 255).astype(np.uint8)
        # h, w = image_np.shape[:2]

        # # Apply mask if provided
        # if mask is not None:
            # mask_tensor = mask[0].cpu().numpy()  # (H, W)
            # mask_binary = (mask_tensor > 0.5).astype(np.uint8)
        # else:
            # mask_binary = np.ones((h, w), dtype=np.uint8)

        # # Convert to grayscale for BRISQUE (which expects float grayscale 0-1)
        # gray_float = rgb2gray(image_np)  # (H, W) float64
        # print("GRAY:", gray_float.shape, gray_float.dtype, np.min(gray_float), np.max(gray_float))
        # # BRISQUE (always on full image)
        # try:
            # if gray_float.ndim == 2 and gray_float.shape[0] > 32 and gray_float.shape[1] > 32:
                # brisque_score = float(brisque.score(gray_float))
            # else:
                # raise ValueError("Image too small or invalid for BRISQUE")
        # except Exception as e:
            # print(f"[BRISQUE ERROR] {e}")
            # brisque_score = -1.0

        # # Blur value (masked)
        # gray_uint8 = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)  # uint8
        # lap = cv2.Laplacian(gray_uint8, cv2.CV_64F)
        # blur_value = float(np.var(lap[mask_binary == 1]))

        # # SNR (masked)
        # mean = cv2.blur(gray_uint8.astype(np.float32), (3, 3))
        # diff = gray_uint8.astype(np.float32) - mean
        # signal_power = np.mean((gray_uint8[mask_binary == 1]) ** 2)
        # noise_power = np.mean((diff[mask_binary == 1]) ** 2)
        # snr_value = float(10 * np.log10(signal_power / noise_power)) if noise_power != 0 else float("inf")

        # # Compression ratio estimation (JPEG)
        # try:
            # pil_image = Image.fromarray(image_np)
            # from io import BytesIO
            # buf = BytesIO()
            # pil_image.save(buf, format="JPEG", quality=95)
            # jpeg_size_kb = len(buf.getvalue()) / 1024
            # raw_size_kb = (h * w * 3) / 1024
            # compression_ratio = float(raw_size_kb / jpeg_size_kb) if jpeg_size_kb > 0 else 0.0
        # except:
            # compression_ratio = 1.0

        # # grayscale float image from rgb2gray: shape (H, W), range [0.0, 1.0]
        # gray_preview = np.clip(gray_float, 0.0, 1.0).astype(np.float32)

        # # replicate into 3 channels (H, W) → (H, W, 3)
        # gray_rgb = np.stack([gray_preview] * 3, axis=-1)

        # # convert to tensor (1, 3, H, W)
        # gray_tensor = torch.from_numpy(gray_rgb).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

        # return (
            # gray_tensor,
            # round(brisque_score, 2),
            # round(blur_value, 2),
            # round(snr_value, 2),
            # round(compression_ratio, 2),
            # HELP_MESSAGE
        # )

# # ComfyUI node registration
# NODE_CLASS_MAPPINGS = {
    # "FRED_ImageQualityInspector": FRED_ImageQualityInspector
# }

# NODE_DISPLAY_NAME_MAPPINGS = {
    # "FRED_ImageQualityInspector": "👑 FRED Image Quality Inspector"
# }

import torch
import numpy as np
import cv2
from PIL import Image
from piq import brisque
from skimage import img_as_float32

HELP_MESSAGE = """
⭐️ Image Quality Analysis (BRISQUE, Blur, SNR, Compression) ⭐️

• **BRISQUE Score (lower is better)**
    - <20 = parfait | 20–30 = très bon | 30–40 = OK | >40 = douteux
    - Plus la valeur est BASSE, meilleure est la qualité perçue.
    - BRISQUE : Blind/Referenceless Image Spatial Quality Evaluator (no ref).

• **Blur Value (0–500+)**
    - Laplacian variance (sur mask si fourni)
    - >300 : très net | 100–300 : net | <100 : flou probable

• **SNR (Signal/Noise Ratio, dB)**
    - >25 : très peu de bruit | 15–25 : bruit modéré | <15 : bruité

• **Compression Ratio (Raw/JPEG)**
    - ~1.0 : normal | <0.7 : compression forte | >1.2 : très haute qualité

BRISQUE doc : https://piq.readthedocs.io/en/latest/metrics.html#brisque
"""

class FRED_ImageQualityInspector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",)},
            "optional": {"mask": ("MASK",)},
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("BRISQUE_SCORE", "BLUR_VALUE", "SNR_VALUE", "COMPRESSION_RATIO", "help")
    FUNCTION = "analyze_image"
    CATEGORY = "👑FRED/analysis"

    def analyze_image(self, image, mask=None):
        # Convert ComfyUI image to numpy
        image_tensor = image[0]  # (C, H, W)
        image_np = image_tensor.cpu().numpy()
        if image_np.shape[0] == 3:
            image_np = image_np.transpose(1, 2, 0)  # (H, W, C)
        elif image_np.shape[2] == 3:
            pass
        else:
            raise ValueError(f"Unexpected image shape: {image_np.shape}")
        image_np = np.clip(image_np, 0, 1)
        h, w = image_np.shape[:2]

        # BRISQUE: input (H,W,3), float32, [0,1]
        image_brisque = img_as_float32(image_np)
        # Doit être (1, 3, H, W) et torch.float32, device cpu
        if image_brisque.shape[-1] == 3:  # (H,W,3)
            image_brisque = torch.from_numpy(image_brisque).permute(2,0,1).unsqueeze(0).float()
        else:
            # Si déjà torch et bonne shape
            image_brisque = torch.from_numpy(image_brisque).unsqueeze(0).float()
        brisque_score = float(brisque(image_brisque, data_range=1.0).item())

        # Mask pour blur/SNR
        if mask is not None:
            mask_tensor = mask[0].cpu().numpy()
            mask_binary = (mask_tensor > 0.5).astype(np.uint8)
        else:
            mask_binary = np.ones((h, w), dtype=np.uint8)

        # Blur Value (Laplacian variance, sur mask)
        image_uint8 = (image_np * 255).astype(np.uint8)
        gray_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(gray_uint8, cv2.CV_64F)
        blur_value = float(np.var(lap[mask_binary == 1]))

        # SNR (Signal to Noise Ratio, sur mask)
        mean = cv2.blur(gray_uint8.astype(np.float32), (3, 3))
        diff = gray_uint8.astype(np.float32) - mean
        signal_power = np.mean((gray_uint8[mask_binary == 1]) ** 2)
        noise_power = np.mean((diff[mask_binary == 1]) ** 2)
        snr_value = float(10 * np.log10(signal_power / noise_power)) if noise_power != 0 else float("inf")

        # Compression Ratio (estimation JPEG vs raw)
        try:
            pil_img = Image.fromarray(image_uint8)
            from io import BytesIO
            buf = BytesIO()
            pil_img.save(buf, format="JPEG", quality=95)
            jpeg_size_kb = len(buf.getvalue()) / 1024
            raw_size_kb = (h * w * 3) / 1024
            compression_ratio = float(raw_size_kb / jpeg_size_kb) if jpeg_size_kb > 0 else 0.0
        except:
            compression_ratio = 1.0

        return (
            round(brisque_score, 3),
            round(blur_value, 2),
            round(snr_value, 2),
            round(compression_ratio, 2),
            HELP_MESSAGE
        )

NODE_CLASS_MAPPINGS = {
    "FRED_ImageQualityInspector": FRED_ImageQualityInspector
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_ImageQualityInspector": "👑 FRED Image Quality Inspector (BRISQUE)"
}