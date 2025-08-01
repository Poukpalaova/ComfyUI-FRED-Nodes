import cv2
from PIL import Image
import numpy as np
import torch
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
import comfy.utils
from comfy_extras.nodes_mask import ImageCompositeMasked

ASPECT_RATIOS = [
    {"name": "9:21 portrait 640x1536", "width": 640, "height": 1536},
    {"name": "1:2 portrait 768x1536", "width": 768, "height": 1536},
    {"name": "9:16 portrait 768x1344", "width": 768, "height": 1344},
    {"name": "2:3 portrait 1024x1536", "width": 1024, "height": 1536},
    {"name": "5:8 portrait 832x1216", "width": 832, "height": 1216},
    {"name": "5:7 portrait 896x1254", "width": 896, "height": 1254},
    {"name": "3:4 portrait 896x1152", "width": 896, "height": 1152},
    {"name": "4:5 portrait 1024x1280", "width": 1024, "height": 1280},
    {"name": "5:6 portrait 1066x1280", "width": 1066, "height": 1280},
    {"name": "9:10 portrait 1152x1280", "width": 1152, "height": 1280},
    {"name": "1:1 square 1024x1024", "width": 1024, "height": 1024},
    {"name": "10:9 landscape 1280x1152", "width": 1280, "height": 1152},
    {"name": "6:5 landscape 1280x1066", "width": 1280, "height": 1066},
    {"name": "5:4 landscape 1280x1024", "width": 1280, "height": 1024},
    {"name": "8:5 landscape 1280x800", "width": 1280, "height": 800},
    {"name": "7:5 landscape 1120x800", "width": 1120, "height": 800},
    {"name": "4:3 landscape 1152x896", "width": 1152, "height": 896},
    {"name": "3:2 landscape 1216x832", "width": 1216, "height": 832},
    {"name": "16:9 wide landscape 1344x768", "width": 1344, "height": 768},
    {"name": "2:1 panorama 1536x768", "width": 1536, "height": 768},
    {"name": "21:9 ultra-wide 1536x640", "width": 1536, "height": 640}
]

HELP_MESSAGE = """This node automatically crops and resizes images to fit aspect ratios.

Key features:
1. Auto-find resolution: Set to True to automatically find the closest ratio for your image.
2. Custom aspect ratios: Choose from predefined ratios or set a custom width and height.
3. Cropping options:
- Crop from center or adjust using crop_x_in_Percent and crop_y_in_Percent.
- Option to pre-crop based on an input mask.
4. Resizing:
- Option to resize the cropped image to the target dimensions.
- Different interpolation modes for upscaling and downscaling.
5. Prescaling: Apply a prescale factor to increase or decrease the final image size.
6. Preview: Generates a preview image showing the cropped area.
7. Mask handling: Can process and modify input masks alongside the image.

Use 'Auto_find_resolution' for automatic ratio selection, or choose a specific ratio.
Adjust cropping, resizing, and scaling options to fine-tune the output.
The node provides both the processed image and a visual preview of the changes.

New: 'Auto_find_resolution_mask_preserve' will ensure the mask area is never cropped, shifting the crop as needed.
"""

class FRED_AutoCropImage_Native_Ratio_v5:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "Precrop_from_input_mask": ("BOOLEAN", {"default": False}),
                "aspect_ratio": (
                    ["custom", "Auto_find_resolution", "Auto_find_resolution_mask_preserve"] +
                    [aspect_ratio["name"] for aspect_ratio in ASPECT_RATIOS] +
                    ["no_crop_to_ratio"], {"default": "Auto_find_resolution"}
                ),
                "custom_width": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "custom_height": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "crop_from_center": ("BOOLEAN", {"default": True}),
                "crop_x_in_Percent": ("INT", {"default": 0, "min": 0, "max": 100}),
                "crop_y_in_Percent": ("INT", {"default": 0, "min": 0, "max": 100}),
                "resize_image": ("BOOLEAN", {"default": False}),
                "resize_mode_if_upscale": (["bicubic", "bilinear", "nearest", "nearest-exact", "area"], {"default": "bilinear"}),
                "resize_mode_if_downscale": (["bicubic", "bilinear", "nearest", "nearest-exact", "area"], {"default": "area"}),
                "prescale_factor": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 8.0, "step": 0.1}),
                "include_prescale_if_resize": ("BOOLEAN", {"default": False}),
                "multiple_of": (["1", "2", "4", "8", "16", "32", "64"], {"default": "1"}),
                "preview_mask_color_intensity": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0, "step": 0.1}),
                "preview_mask_color": ("COLOR", {"default": "#503555"},),
            },
            "optional": {
                "mask_optional": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "FLOAT", "INT", "INT", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = (
        "modified_image",
        "preview",
        "modified_mask",
        "scale_factor",
        "output_width",
        "output_height",
        "native_width",
        "native_height",
        "sd_aspect_ratios",
        "help"
    )
    FUNCTION = "run"
    CATEGORY = "👑FRED/image/postprocessing"
    OUTPUT_NODE = True

    def run(
        self, image, Precrop_from_input_mask, aspect_ratio, custom_width, custom_height,
        crop_from_center, crop_x_in_Percent, crop_y_in_Percent, resize_image,
        resize_mode_if_upscale, resize_mode_if_downscale, prescale_factor,
        include_prescale_if_resize, preview_mask_color_intensity, preview_mask_color, multiple_of, mask_optional=None
    ):
        _, original_height, original_width, _ = image.shape

        # Gestion du mask en entrée (redimensionnement au besoin)
        if mask_optional is None:
            mask = torch.zeros(1, original_height, original_width, dtype=torch.float32, device=image.device)
        else:
            mask = mask_optional
        if mask.shape[1] != original_height or mask.shape[2] != original_width:
            mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=(original_height, original_width), mode="bicubic").squeeze(0).clamp(0.0, 1.0)

        # Precrop autour du mask si demandé
        if Precrop_from_input_mask and mask_optional is not None:
            x_min, y_min, x_max, y_max = self.find_mask_boundaries(mask)
            if x_min is not None:
                image = image[:, y_min:y_max+1, x_min:x_max+1, :]
                mask = mask[:, y_min:y_max+1, x_min:x_max+1]
                _, original_height, original_width, _ = image.shape

        # Modes principaux
        if aspect_ratio == "no_crop_to_ratio":
            preview = image
            cropped_image = image
            cropped_mask = mask
            native_width = original_width
            native_height = original_height
            sd_aspect_ratios = "no_crop"

        elif aspect_ratio == "Auto_find_resolution_mask_preserve":
            if mask is not None and mask.sum() > 0:
                x_min, y_min, x_max, y_max = self.find_mask_boundaries(mask)
                print(f"[DEBUG MASK BOUNDARIES] x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}, mask_w={x_max-x_min+1}, mask_h={y_max-y_min+1}")
                mask_w = x_max - x_min + 1
                mask_h = y_max - y_min + 1

                candidates = []
                print(f"[MASK BBOX] x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}, mask_w={mask_w}, mask_h={mask_h}")
                for entry in ASPECT_RATIOS:
                    native_width, native_height = entry["width"], entry["height"]
                    # Cherche le scale maximal du ratio dans l'image qui peut couvrir le mask
                    scale_w = original_width / native_width
                    scale_h = original_height / native_height
                    scale_for_mask_w = mask_w / native_width
                    scale_for_mask_h = mask_h / native_height

                    scale_min = max(scale_for_mask_w, scale_for_mask_h)
                    scale_max = min(scale_w, scale_h)

                    if scale_min > scale_max:
                        continue

                    scale = scale_max
                    crop_w = int(native_width * scale)
                    crop_h = int(native_height * scale)

                    # Position de crop (préférence utilisateur)
                    if crop_from_center:
                        mask_center_x = (x_min + x_max) // 2
                        mask_center_y = (y_min + y_max) // 2
                        ideal_x = mask_center_x - crop_w // 2
                        ideal_y = mask_center_y - crop_h // 2

                        print(f"[DEBUG CALCUL CENTER] mask_center_x={mask_center_x}, mask_center_y={mask_center_y}, crop_w={crop_w}, crop_h={crop_h}, ideal_x={ideal_x}, ideal_y={ideal_y}")

                        x_start = int(np.clip(ideal_x, 0, original_width - crop_w))
                        y_start = int(np.clip(ideal_y, 0, original_height - crop_h))

                        print(f"[DEBUG AFTER CLAMP] x_start={x_start}, y_start={y_start}, original_width={original_width}, original_height={original_height}")
                    else:
                        x_start = int((crop_x_in_Percent / 100) * (original_width - crop_w))
                        y_start = int((crop_y_in_Percent / 100) * (original_height - crop_h))

                    # Coller au(x) bord(s) si le mask les touche
                    if y_min == 0:
                        y_start = 0
                    if y_max == original_height - 1:
                        y_start = original_height - crop_h
                    if x_min == 0:
                        x_start = 0
                    if x_max == original_width - 1:
                        x_start = original_width - crop_w

                    # Clamp final
                    x_start = max(0, min(x_start, original_width - crop_w))
                    y_start = max(0, min(y_start, original_height - crop_h))

                    crop_x_min = x_start
                    crop_x_max = x_start + crop_w - 1
                    crop_y_min = y_start
                    crop_y_max = y_start + crop_h - 1

                    # Vérifie que le crop englobe le mask
                    mask_in_crop = (crop_x_min <= x_min and crop_x_max >= x_max and
                                    crop_y_min <= y_min and crop_y_max >= y_max)

                    if mask_in_crop:
                        area = crop_w * crop_h
                        candidates.append((area, crop_w, crop_h, x_start, y_start, entry["name"], entry["width"], entry["height"]))

                if candidates:
                    # Prend le meilleur ratio par surface cropable, puis recalcule la position pour la taille SD arrondie
                    _, _, _, _, _, sd_aspect_ratios, native_width_orig, native_height_orig = max(candidates, key=lambda tup: tup[0])
                    m = int(multiple_of)
                    native_width = (native_width_orig // m) * m
                    native_height = (native_height_orig // m) * m

                    print(f"[DEBUG] Meilleur ratio: {sd_aspect_ratios}, native_width arrondi={native_width}, native_height arrondi={native_height}")

                    # --- RECALCULE position pour la taille finale (native_width/native_height) ---
                    if crop_from_center:
                        ideal_x = (x_min + x_max) // 2 - native_width // 2
                        ideal_y = (y_min + y_max) // 2 - native_height // 2
                        x_start = int(np.clip(ideal_x, 0, original_width - native_width))
                        y_start = int(np.clip(ideal_y, 0, original_height - native_height))
                    else:
                        x_start = int((crop_x_in_Percent / 100) * (original_width - native_width))
                        y_start = int((crop_y_in_Percent / 100) * (original_height - native_height))

                    if y_min == 0:
                        y_start = 0
                    if y_max == original_height - 1:
                        y_start = original_height - native_height
                    if x_min == 0:
                        x_start = 0
                    if x_max == original_width - 1:
                        x_start = original_width - native_width

                    x_start = max(0, min(x_start, original_width - native_width))
                    y_start = max(0, min(y_start, original_height - native_height))

                    print(f"[DEBUG FINAL CROP] x_start={x_start}, y_start={y_start}, native_width={native_width}, native_height={native_height}")

                    new_crop_x_in_Percent = int(100 * x_start / max(1, original_width - native_width))
                    new_crop_y_in_Percent = int(100 * y_start / max(1, original_height - native_height))

                    cropped_image, preview = self.crop_image_to_ratio(
                        image, native_width, native_height,
                        False,
                        new_crop_x_in_Percent, new_crop_y_in_Percent,
                        False, preview_mask_color_intensity, preview_mask_color
                    )
                    cropped_mask = self.crop_image_to_ratio(
                        mask, native_width, native_height,
                        False,
                        new_crop_x_in_Percent, new_crop_y_in_Percent,
                        True, preview_mask_color_intensity, preview_mask_color
                    )
                else:
                    print("finding crop resolution and position to preserve the mask failled, fallback to auto find")
                    # Fallback : utilise la logique existante
                    native_width, native_height, sd_aspect_ratios = self.find_closest_sd_resolution(original_width, original_height)
                    m = int(multiple_of)
                    native_width = (native_width // m) * m
                    native_height = (native_height // m) * m
                    cropped_image, preview = self.crop_image_to_ratio(
                        image, native_width, native_height,
                        crop_from_center, crop_x_in_Percent, crop_y_in_Percent,
                        False, preview_mask_color_intensity, preview_mask_color
                    )
                    cropped_mask = self.crop_image_to_ratio(
                        mask, native_width, native_height,
                        crop_from_center, crop_x_in_Percent, crop_y_in_Percent,
                        True, preview_mask_color_intensity, preview_mask_color
                    )

            else:
                native_width, native_height, sd_aspect_ratios = self.find_closest_sd_resolution(original_width, original_height)
                m = int(multiple_of)
                native_width = (native_width // m) * m
                native_height = (native_height // m) * m
                cropped_image, preview = self.crop_image_to_ratio(image, native_width, native_height, crop_from_center, crop_x_in_Percent, crop_y_in_Percent, False, preview_mask_color_intensity, preview_mask_color)
                cropped_mask = self.crop_image_to_ratio(mask, native_width, native_height, crop_from_center, crop_x_in_Percent, crop_y_in_Percent, True, preview_mask_color_intensity, preview_mask_color)

        elif aspect_ratio == "Auto_find_resolution":
            native_width, native_height, sd_aspect_ratios = self.find_closest_sd_resolution(original_width, original_height)
            m = int(multiple_of)
            native_width = (native_width // m) * m
            native_height = (native_height // m) * m
            cropped_image, preview = self.crop_image_to_ratio(image, native_width, native_height, crop_from_center, crop_x_in_Percent, crop_y_in_Percent, False, preview_mask_color_intensity, preview_mask_color)
            cropped_mask = self.crop_image_to_ratio(mask, native_width, native_height, crop_from_center, crop_x_in_Percent, crop_y_in_Percent, True, preview_mask_color_intensity, preview_mask_color)

        elif aspect_ratio == "custom":
            native_width = custom_width
            native_height = custom_height
            m = int(multiple_of)
            native_width = (native_width // m) * m
            native_height = (native_height // m) * m
            cropped_image, preview = self.crop_image_to_ratio(image, native_width, native_height, crop_from_center, crop_x_in_Percent, crop_y_in_Percent, False, preview_mask_color_intensity, preview_mask_color)
            cropped_mask = self.crop_image_to_ratio(mask, native_width, native_height, crop_from_center, crop_x_in_Percent, crop_y_in_Percent, True, preview_mask_color_intensity, preview_mask_color)

        else:
            ratio = next(a for a in ASPECT_RATIOS if a["name"] == aspect_ratio)
            native_width, native_height = ratio["width"], ratio["height"]
            m = int(multiple_of)
            native_width = (native_width // m) * m
            native_height = (native_height // m) * m
            cropped_image, preview = self.crop_image_to_ratio(image, native_width, native_height, crop_from_center, crop_x_in_Percent, crop_y_in_Percent, False, preview_mask_color_intensity, preview_mask_color)
            cropped_mask = self.crop_image_to_ratio(mask, native_width, native_height, crop_from_center, crop_x_in_Percent, crop_y_in_Percent, True, preview_mask_color_intensity, preview_mask_color)

        _, cropped_height, cropped_width, _ = cropped_image.shape

        # --- Resize si demandé (logique universelle pour tous les modes)
        if resize_image:
            m = int(multiple_of)
            if include_prescale_if_resize:
                final_width = int(((native_width * prescale_factor) // m) * m)
                final_height = int(((native_height * prescale_factor) // m) * m)
                scale_factor = prescale_factor
            else:
                final_width = (native_width // m) * m
                final_height = (native_height // m) * m
                # scale_factor = final_width / cropped_width
                scale_factor = min(final_width / cropped_width, final_height / cropped_height)
            resize_interpolation_mode = resize_mode_if_downscale if final_width < cropped_width or final_height < cropped_height else resize_mode_if_upscale
            modified_image = self.resize_image(cropped_image, resize_interpolation_mode, final_width, final_height, "center")
            modified_mask = comfy.utils.common_upscale(cropped_mask.unsqueeze(1), final_width, final_height, resize_interpolation_mode, "center").squeeze(1)
        else:
            # scale_factor = native_width / cropped_width
            scale_factor = min(native_width / cropped_width, native_height / cropped_height)
            modified_image = cropped_image
            modified_mask = cropped_mask

        if 'sd_aspect_ratios' not in locals():
            sd_aspect_ratios = aspect_ratio

        _, output_height, output_width, _ = modified_image.shape

        # Affiche overlay mask seulement si mask partiel
        if mask is not None and mask.sum() > 0 and mask.sum() != mask.numel():
            # preview = self.apply_mask_overlay(preview, mask, preview_mask_color)
            preview = self.apply_mask_overlay(preview, mask_optional, preview_mask_color, preview_mask_color_intensity)

        return (
            modified_image,
            preview,
            modified_mask,
            scale_factor,
            output_width,
            output_height,
            native_width,
            native_height,
            sd_aspect_ratios,
            HELP_MESSAGE
        )

    # --- Fonctions utilitaires ---

    def apply_mask_overlay(self, preview: torch.Tensor, mask: torch.Tensor, mask_color: str, alpha: float = 0.6) -> torch.Tensor:
        """
        preview: (B,H,W,3) float32 values in [0–1]
        mask:    (B,H,W)   float32 values in [0–1]
        mask_color: HEX string, e.g. "#FF0000"
        alpha: opacity of the overlay
        """

        # Convertir la couleur HEX en RGB float [0–1]
        rgb = torch.tensor(self.Hex_to_RGB(mask_color), dtype=preview.dtype, device=preview.device) / 255.0
        B, H, W, C = preview.shape

        # S’assurer que mask est (B,H,W,1)
        if mask.ndim == 3:
            mask = mask.unsqueeze(-1)  # (B,H,W,1)

        # Si le mask ne correspond pas à la taille du preview, le redimensionner
        if mask.shape[1:3] != (H, W):
            # mask: (B,1,H0,W0) -> permute en (B,C,H0,W0) pour interpolate
            mask = mask.permute(0, 3, 1, 2)  
            mask = torch.nn.functional.interpolate(mask, size=(H, W), mode="nearest")
            mask = mask.permute(0, 2, 3, 1)  # (B,H,W,1)

        mask = mask.clamp(0.0, 1.0)

        # Construire l’overlay (B,H,W,3)
        overlay = rgb.view(1, 1, 1, 3).expand(B, H, W, 3)

        # Mélanger preview et overlay uniquement là où mask == 1
        #    blended = preview * (1 - mask*alpha) + overlay * (mask*alpha)
        blended = preview * (1.0 - mask * alpha) + overlay * (mask * alpha)

        # S’assurer de rester en [0–1]
        return blended.clamp(0.0, 1.0)

    # def draw_preview_box(self, image, x_start, y_start, crop_width, crop_height, preview_mask_color):
        # preview = image.clone()
        # color = torch.tensor(self.Hex_to_RGB(preview_mask_color), dtype=torch.uint8, device=image.device)
        # # Draw rectangle (top, bottom, left, right)
        # preview[:, y_start:y_start+crop_height, x_start, :] = color
        # preview[:, y_start:y_start+crop_height, x_start+crop_width-1, :] = color
        # preview[:, y_start, x_start:x_start+crop_width, :] = color
        # preview[:, y_start+crop_height-1, x_start:x_start+crop_width, :] = color
        # return preview

    def find_mask_boundaries(self, mask):
        if mask is None:
            return None, None, None, None
        mask_np = mask.squeeze().cpu().numpy()
        rows = np.any(mask_np, axis=1)
        cols = np.any(mask_np, axis=0)
        if not np.any(rows) or not np.any(cols):
            return 0, 0, mask_np.shape[1] - 1, mask_np.shape[0] - 1
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        return x_min, y_min, x_max, y_max

    def resize_image(self, cropped_image, resize_interpolation_mode, width, height, crop_from_center):
        samples = cropped_image.movedim(-1, 1)
        resized_image = comfy.utils.common_upscale(samples, width, height, resize_interpolation_mode, crop_from_center)
        resized_image = resized_image.movedim(1, -1)
        return resized_image

    def find_closest_sd_resolution(self, original_width, original_height):
        sd_aspect_ratios = [(a["name"], a["width"], a["height"]) for a in ASPECT_RATIOS]
        original_aspect_ratio = original_width / original_height

        # 1. Chercher ratio exactement égal
        for name, native_width, native_height in sd_aspect_ratios:
            if abs((native_width / native_height) - original_aspect_ratio) < 0.001:
                return native_width, native_height, f"{name} - ({native_width}x{native_height})"

        # 2. Sinon, chercher le plus proche comme avant
        closest_distance = float('inf')
        found_native_width = found_native_height = 0
        found_sd_aspect_ratios = None
        for name, native_width, native_height in sd_aspect_ratios:
            sd_aspect_ratio = native_width / native_height
            ratio_distance = abs(original_aspect_ratio - sd_aspect_ratio)
            if ratio_distance < closest_distance:
                closest_distance = ratio_distance
                found_native_width = native_width
                found_native_height = native_height
                found_sd_aspect_ratios = f"{name} - ({native_width}x{native_height})"
        return found_native_width, found_native_height, found_sd_aspect_ratios

    # def find_closest_sd_resolution_preserve_mask(self, original_width, original_height, mask):
        # x_min, y_min, x_max, y_max = self.find_mask_boundaries(mask)
        # mask_w = x_max - x_min + 1
        # mask_h = y_max - y_min + 1
        # original_aspect = original_width / original_height
        # closest_distance = float('inf')
        # chosen_w = chosen_h = None
        # chosen_name = None

        # for entry in ASPECT_RATIOS:
            # sd_w, sd_h = entry["width"], entry["height"]
            # if mask_w <= sd_w and mask_h <= sd_h:
                # sd_aspect = sd_w / sd_h
                # dist = abs(original_aspect - sd_aspect)
                # if dist < closest_distance:
                    # closest_distance = dist
                    # chosen_w, chosen_h = sd_w, sd_h
                    # chosen_name = entry["name"]

        # # Fallback si aucun ratio ne pouvait contenir le masque
        # if chosen_name is None:
            # return self.find_closest_sd_resolution(original_width, original_height)

        # return chosen_w, chosen_h, f"{chosen_name} - ({chosen_w}x{chosen_h})"

    # def find_crop_position_to_preserve_mask(
        # self, mask_bounds, crop_width, crop_height, image_width, image_height, 
        # crop_from_center, crop_x_percent, crop_y_percent
    # ):
        # x_min, y_min, x_max, y_max = mask_bounds
        # mask_w = x_max - x_min + 1
        # mask_h = y_max - y_min + 1
        # # Si le masque dépasse le crop, on force à la bounding box du mask
        # if (mask_w > crop_width) or (mask_h > crop_height):
            # x_start = max(0, min(x_min, image_width - crop_width))
            # y_start = max(0, min(y_min, image_height - crop_height))
            # return x_start, y_start
        # # Centre du masque
        # mask_center_x = (x_min + x_max) // 2
        # mask_center_y = (y_min + y_max) // 2
        # if crop_from_center:
            # x_start = max(0, min(image_width - crop_width, mask_center_x - crop_width // 2))
            # y_start = max(0, min(image_height - crop_height, mask_center_y - crop_height // 2))
        # else:
            # # Nouvelle logique : percent indique où aligner le crop
            # x_start = int((crop_x_percent / 100) * (image_width - crop_width))
            # y_start = int((crop_y_percent / 100) * (image_height - crop_height))
            # # On s'assure que la bbox du mask est toujours incluse
            # x_start = max(x_start, x_max + 1 - crop_width)
            # x_start = min(x_start, x_min)
            # y_start = max(y_start, y_max + 1 - crop_height)
            # y_start = min(y_start, y_min)
            # x_start = max(0, min(x_start, image_width - crop_width))
            # y_start = max(0, min(y_start, image_height - crop_height))
        # return x_start, y_start

    def crop_image_to_ratio(self, image, native_width, native_height, crop_from_center, crop_x_in_Percent, crop_y_in_Percent, is_mask, preview_mask_color_intensity, preview_mask_color):
        if len(image.shape) == 4:
            _, original_height, original_width, _ = image.shape
        elif len(image.shape) == 3:
            _, original_height, original_width = image.shape
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")

        # Calcul de l'aspect ratio cible
        target_aspect_ratio = native_width / native_height

        # Calcule la largeur/hauteur à cropper selon le ratio
        if (original_width / original_height) > target_aspect_ratio:
            new_height = original_height
            new_width = int(new_height * target_aspect_ratio)
        else:
            new_width = original_width
            new_height = int(new_width / target_aspect_ratio)

        # Calcul la position du crop
        if crop_from_center:
            x_start = max(0, (original_width - new_width) // 2)
            y_start = max(0, (original_height - new_height) // 2)
        else:
            x_start = int((crop_x_in_Percent / 100) * (original_width - new_width))
            y_start = int((crop_y_in_Percent / 100) * (original_height - new_height))
            x_start = min(max(0, x_start), original_width - new_width)
            y_start = min(max(0, y_start), original_height - new_height)

        if is_mask:
            cropped_image = image[:, y_start:y_start + new_height, x_start:x_start + new_width]
            return cropped_image
        else:
            # Convertir la couleur HEX en RGB
            preview_color = torch.tensor(self.Hex_to_RGB(preview_mask_color), dtype=torch.uint8, device=image.device)

            # Create a preview base with the original image dimensions
            preview = image.clone() if len(image.shape) == 4 else image.clone().unsqueeze(0)

            # Create a white overlay image
            overlay_image = torch.full((1, original_height, original_width, 3), 255, dtype=torch.uint8, device=image.device)

            if x_start > 0:  # Left uncropped region
                overlay_image[:, :, :x_start, :] = preview_color
            if x_start + new_width < original_width:  # Right uncropped region
                overlay_image[:, :, x_start + new_width:, :] = preview_color

            # Vertical bands (top and bottom)
            if y_start > 0:  # Top uncropped region
                overlay_image[:, :y_start, x_start:x_start+new_width, :] = preview_color
            if y_start + new_height < original_height:  # Bottom uncropped region
                overlay_image[:, y_start + new_height:, x_start:x_start+new_width, :] = preview_color


            # Convert to float and normalize to 0-1 range if needed for further processing
            overlay_image_float = overlay_image.float() / 255.0

            # Make sure preview is also a float tensor
            preview_float = preview.float()
            blend_preview = self.blend_images(preview_float, overlay_image_float, preview_mask_color_intensity)

            # Convertir en numpy array AVANT d'utiliser ascontiguousarray
            blend_preview_np = (blend_preview[0].cpu().numpy() * 255).astype(np.uint8)
            blend_preview_np = np.ascontiguousarray(blend_preview_np)
            
            # Draw rectangle for the cropped region
            cv2.rectangle(blend_preview_np, (x_start, y_start), (x_start + new_width, y_start + new_height), (int(preview_color[0]), int(preview_color[1]), int(preview_color[2])), 4)
            blend_preview = torch.from_numpy(blend_preview_np).unsqueeze(0).float() / 255.0

            # Crop the image
            if len(image.shape) == 4:
                cropped_image = image[:, y_start:y_start + new_height, x_start:x_start + new_width, :]
            elif len(image.shape) == 3:
                cropped_image = image[y_start:y_start + new_height, x_start:x_start + new_width]
            else:
                raise ValueError(f"Unexpected image shape: {image.shape}")

            return cropped_image, blend_preview
        
    def blend_images(self, image1: torch.Tensor, image2: torch.Tensor, blend_factor):
        if image1.shape != image2.shape:
            image2 = self.crop_and_resize(image2, image1.shape)

        blended_image = image1 * image2
        blended_image = image1 * (1 - blend_factor) + blended_image * blend_factor
        blended_image = torch.clamp(blended_image, 0, 1)
        # return (blended_image,)
        return blended_image
        
    def crop_and_resize(self, img: torch.Tensor, target_shape: tuple):
        batch_size, img_h, img_w, img_c = img.shape
        _, target_h, target_w, _ = target_shape
        img_aspect_ratio = img_w / img_h
        target_aspect_ratio = target_w / target_h

        # Crop center of the image to the target aspect ratio
        if img_aspect_ratio > target_aspect_ratio:
            new_width = int(img_h * target_aspect_ratio)
            left = (img_w - new_width) // 2
            img = img[:, :, left:left + new_width, :]
        else:
            new_height = int(img_w / target_aspect_ratio)
            top = (img_h - new_height) // 2
            img = img[:, top:top + new_height, :, :]

        # Resize to target size
        img = img.permute(0, 3, 1, 2) # Torch wants (B, C, H, W) we use (B, H, W, C)
        img = F.interpolate(img, size=(target_h, target_w), mode='bilinear', align_corners=False)
        img = img.permute(0, 2, 3, 1)

        return img
        
    def Hex_to_RGB(self, inhex:str) -> tuple:
        if not inhex.startswith('#'):
            raise ValueError(f'Invalid Hex Code in {inhex}')
        else:
            rval = inhex[1:3]
            gval = inhex[3:5]
            bval = inhex[5:]
            rgb = (int(rval, 16), int(gval, 16), int(bval, 16))
        return tuple(rgb)

    @staticmethod
    def IS_CHANGED(*args, **kwargs):
        return False

# Mappings ComfyUI
NODE_CLASS_MAPPINGS = {
    "FRED_AutoCropImage_Native_Ratio_v5": FRED_AutoCropImage_Native_Ratio_v5
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_AutoCropImage_Native_Ratio_v5": "👑 FRED AutoCropImage Native Ratio v5"
}
