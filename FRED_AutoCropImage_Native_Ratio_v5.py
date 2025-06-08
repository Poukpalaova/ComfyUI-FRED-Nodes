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
    {"name": "5:8 portrait 832x1216", "width": 832, "height": 1216},
    {"name": "3:4 portrait 896x1152", "width": 896, "height": 1152},
    {"name": "4:5 portrait 1024x1280", "width": 1024, "height": 1280},
    {"name": "5:6 portrait 1066x1280", "width": 1066, "height": 1280},
    {"name": "9:10 portrait 1152x1280", "width": 1152, "height": 1280},
    {"name": "1:1 square 1024x1024", "width": 1024, "height": 1024},
    {"name": "10:9 landscape 1280x1152", "width": 1280, "height": 1152},
    {"name": "6:5 landscape 1280x1066", "width": 1280, "height": 1066},
    {"name": "5:4 landscape 1280x1024", "width": 1280, "height": 1024},
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
                "Precrop_from_input_mask": ("BOOLEAN", {"default": False},),
                "aspect_ratio": (["custom"] + ["Auto_find_resolution"] + ["Auto_find_resolution_mask_preserve"] + [aspect_ratio["name"] for aspect_ratio in ASPECT_RATIOS] + ["no_crop_to_ratio"],),
                "custom_width": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "custom_height": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "crop_from_center": ("BOOLEAN", {"default": True},),
                "crop_x_in_Percent": ("INT", {"default": 0, "min": 0, "max": 100}),
                "crop_y_in_Percent": ("INT", {"default": 0, "min": 0, "max": 100}),
                "resize_image": ("BOOLEAN", {"default": False},),
                "resize_mode_if_upscale": (["bicubic", "bilinear", "nearest", "nearest-exact", "area"], {"default": "bilinear"}),
                "resize_mode_if_downscale": (["bicubic", "bilinear", "nearest", "nearest-exact", "area"], {"default": "area"}),
                "prescale_factor": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 8.0, "step": 0.1}),
                "include_prescale_if_resize": ("BOOLEAN", {"default": False},),
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
    CATEGORY = "FRED/image/postprocessing"
    OUTPUT_NODE = True

    def run(
        self, image, Precrop_from_input_mask, aspect_ratio, custom_width, custom_height,
        crop_from_center, crop_x_in_Percent, crop_y_in_Percent, resize_image,
        resize_mode_if_upscale, resize_mode_if_downscale, prescale_factor,
        include_prescale_if_resize, preview_mask_color, mask_optional=None
    ):
        _, original_height, original_width, _ = image.shape

        if mask_optional is None:
            mask = torch.zeros(1, original_height, original_width, dtype=torch.float32, device=image.device)
        else:
            mask = mask_optional
        if mask.shape[1] != original_height or mask.shape[2] != original_width:
            mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=(original_height, original_width), mode="bicubic").squeeze(0).clamp(0.0, 1.0)

        if Precrop_from_input_mask and mask_optional is not None:
            x_min, y_min, x_max, y_max = self.find_mask_boundaries(mask)
            if x_min is not None:
                image = image[:, y_min:y_max+1, x_min:x_max+1, :]
                mask = mask[:, y_min:y_max+1, x_min:x_max+1]
                _, original_height, original_width, _ = image.shape

        if aspect_ratio == "no_crop_to_ratio":
            preview = image
            cropped_image = image
            cropped_mask = mask
            native_width = original_width
            native_height = original_height

        elif aspect_ratio == "Auto_find_resolution_mask_preserve":
            if mask is not None and mask.sum() > 0:
                # 1. Find best SD resolution for mask
                native_width, native_height, sd_aspect_ratios = self.find_closest_sd_resolution_preserve_mask(
                    original_width, original_height, mask
                )
                # 2. Find crop position so mask is never cropped
                x_min, y_min, x_max, y_max = self.find_mask_boundaries(mask)
                crop_width = native_width
                crop_height = native_height
                image_width = original_width
                image_height = original_height

                x_start, y_start = self.find_crop_position_to_preserve_mask(
                    (x_min, y_min, x_max, y_max),
                    crop_width, crop_height, image_width, image_height,
                    crop_from_center, crop_x_in_Percent, crop_y_in_Percent
                )
                # x_start = min(max(x_start, 0), image_width - crop_width)
                # y_start = min(max(y_start, 0), image_height - crop_height)

                # 3. Crop image and mask
                cropped_image = image[:, y_start:y_start+crop_height, x_start:x_start+crop_width, :]
                cropped_mask = mask[:, y_start:y_start+crop_height, x_start:x_start+crop_width]
                preview = self.draw_preview_box(image, x_start, y_start, crop_width, crop_height, preview_mask_color)
            else:
                # fallback to normal auto-find
                native_width, native_height, sd_aspect_ratios = self.find_closest_sd_resolution(original_width, original_height)
                cropped_image, preview = self.crop_image_to_ratio(image, native_width, native_height, crop_from_center, crop_x_in_Percent, crop_y_in_Percent, False, preview_mask_color)
                cropped_mask = self.crop_image_to_ratio(mask, native_width, native_height, crop_from_center, crop_x_in_Percent, crop_y_in_Percent, True, preview_mask_color)

        elif aspect_ratio == "Auto_find_resolution":
            native_width, native_height, sd_aspect_ratios = self.find_closest_sd_resolution(original_width, original_height)
            cropped_image, preview = self.crop_image_to_ratio(image, native_width, native_height, crop_from_center, crop_x_in_Percent, crop_y_in_Percent, False, preview_mask_color)
            cropped_mask = self.crop_image_to_ratio(mask, native_width, native_height, crop_from_center, crop_x_in_Percent, crop_y_in_Percent, True, preview_mask_color)

        elif aspect_ratio == "custom":
            native_width = custom_width
            native_height = custom_height
            cropped_image, preview = self.crop_image_to_ratio(image, native_width, native_height, crop_from_center, crop_x_in_Percent, crop_y_in_Percent, False, preview_mask_color)
            cropped_mask = self.crop_image_to_ratio(mask, native_width, native_height, crop_from_center, crop_x_in_Percent, crop_y_in_Percent, True, preview_mask_color)

        else:
            native_width, native_height = [(a["width"], a["height"]) for a in ASPECT_RATIOS if a["name"] == aspect_ratio][0]
            cropped_image, preview = self.crop_image_to_ratio(image, native_width, native_height, crop_from_center, crop_x_in_Percent, crop_y_in_Percent, False, preview_mask_color)
            cropped_mask = self.crop_image_to_ratio(mask, native_width, native_height, crop_from_center, crop_x_in_Percent, crop_y_in_Percent, True, preview_mask_color)

        _, cropped_height, cropped_width, _ = cropped_image.shape

        if resize_image:
            crop_from_center_str = "center" if crop_from_center else "disabled"
            if include_prescale_if_resize:
                native_width_wfactor = int(native_width * prescale_factor)
                native_height_wfactor = int(native_height * prescale_factor)
                resize_interpolation_mode = resize_mode_if_downscale if native_width_wfactor < cropped_width else resize_mode_if_upscale
                scale_factor = prescale_factor
                resized_image = self.resize_image(cropped_image, resize_interpolation_mode, native_width_wfactor, native_height_wfactor, crop_from_center_str)
                resized_mask = comfy.utils.common_upscale(cropped_mask.unsqueeze(1), native_width_wfactor, native_height_wfactor, resize_interpolation_mode, crop_from_center_str).squeeze(1)
            else:
                resize_interpolation_mode = resize_mode_if_downscale if native_width < cropped_width else resize_mode_if_upscale
                scale_factor = prescale_factor
                resized_image = self.resize_image(cropped_image, resize_interpolation_mode, native_width, native_height, crop_from_center_str)
                resized_mask = comfy.utils.common_upscale(cropped_mask.unsqueeze(1), native_width, native_height, resize_interpolation_mode, crop_from_center_str).squeeze(1)
            modified_image = resized_image
            modified_mask = resized_mask
        else:
            scale_factor = prescale_factor * min(native_width / cropped_width, native_height / cropped_height)
            modified_image = cropped_image
            modified_mask = cropped_mask

        if 'sd_aspect_ratios' not in locals():
            sd_aspect_ratios = aspect_ratio

        _, output_height, output_width, _ = modified_image.shape

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

    def draw_preview_box(self, image, x_start, y_start, crop_width, crop_height, preview_mask_color):
        preview = image.clone()
        color = torch.tensor(self.Hex_to_RGB(preview_mask_color), dtype=torch.uint8, device=image.device)
        # Draw rectangle (top, bottom, left, right)
        preview[:, y_start:y_start+crop_height, x_start, :] = color
        preview[:, y_start:y_start+crop_height, x_start+crop_width-1, :] = color
        preview[:, y_start, x_start:x_start+crop_width, :] = color
        preview[:, y_start+crop_height-1, x_start:x_start+crop_width, :] = color
        return preview

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
        closest_distance = float('inf')
        found_native_width = found_native_height = 0
        found_sd_aspect_ratios = None
        for name, native_width, native_height in sd_aspect_ratios:
            sd_aspect_ratio = native_width / native_height
            ratio_distance = abs(original_aspect_ratio - sd_aspect_ratio)
            dimension_distance = (abs(original_width - native_width) + abs(original_height - native_height)) / (original_width + original_height)
            ratio_weight = 0.5
            dimension_weight = 0.5
            distance = (ratio_weight * ratio_distance) + (dimension_weight * dimension_distance)
            if distance < closest_distance:
                closest_distance = distance
                found_native_width = native_width
                found_native_height = native_height
                found_sd_aspect_ratios = f"{name} - ({native_width}x{native_height})"
        return found_native_width, found_native_height, found_sd_aspect_ratios

    def find_closest_sd_resolution_preserve_mask(self, original_width, original_height, mask):
        x_min, y_min, x_max, y_max = self.find_mask_boundaries(mask)
        mask_w = x_max - x_min + 1
        mask_h = y_max - y_min + 1
        mask_aspect = mask_w / mask_h
        sd_aspect_ratios = [(a["name"], a["width"], a["height"]) for a in ASPECT_RATIOS]
        closest_distance = float('inf')
        found_native_width = found_native_height = 0
        found_sd_aspect_ratios = None
        for name, sd_w, sd_h in sd_aspect_ratios:
            scale = min(sd_w / mask_w, sd_h / mask_h)
            if (mask_w * scale <= sd_w) and (mask_h * scale <= sd_h):
                sd_aspect = sd_w / sd_h
                ratio_distance = abs(mask_aspect - sd_aspect)
                if ratio_distance < closest_distance:
                    closest_distance = ratio_distance
                    found_native_width = sd_w
                    found_native_height = sd_h
                    found_sd_aspect_ratios = f"{name} - ({sd_w}x{sd_h})"
        if found_sd_aspect_ratios is None:
            return self.find_closest_sd_resolution(original_width, original_height)
        return found_native_width, found_native_height, found_sd_aspect_ratios

    def find_crop_position_to_preserve_mask(
        self, mask_bounds, crop_width, crop_height, image_width, image_height, 
        crop_from_center, crop_x_percent, crop_y_percent
    ):
        x_min, y_min, x_max, y_max = mask_bounds
        mask_w = x_max - x_min + 1
        mask_h = y_max - y_min + 1
        # If mask is larger than crop, force to mask bounds (will crop mask)
        if (mask_w > crop_width) or (mask_h > crop_height):
            x_start = max(0, min(x_min, image_width - crop_width))
            y_start = max(0, min(y_min, image_height - crop_height))
            return x_start, y_start
        # Center of the mask bounding box
        mask_center_x = (x_min + x_max) // 2
        mask_center_y = (y_min + y_max) // 2
        if crop_from_center:
            x_start = max(0, min(image_width - crop_width, mask_center_x - crop_width // 2))
            y_start = max(0, min(image_height - crop_height, mask_center_y - crop_height // 2))
        else:
            x_start = round((crop_x_percent * image_width) / 100)
            y_start = round((crop_y_percent * image_height) / 100)
            x_start = max(x_start, x_max + 1 - crop_width)
            x_start = min(x_start, x_min)
            y_start = max(y_start, y_max + 1 - crop_height)
            y_start = min(y_start, y_min)
            x_start = max(0, min(x_start, image_width - crop_width))
            y_start = max(0, min(y_start, image_height - crop_height))
        return x_start, y_start

    def crop_image_to_ratio(self, image, native_width, native_height, crop_from_center, crop_x_in_Percent, crop_y_in_Percent, is_mask, preview_mask_color):
        if len(image.shape) == 4:
            _, original_height, original_width, _ = image.shape
        elif len(image.shape) == 3:
            _, original_height, original_width = image.shape
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")

        crop_x = round((crop_x_in_Percent * original_width) / 100)
        crop_y = round((crop_y_in_Percent * original_height) / 100)

        # Calculate the target aspect ratio
        target_aspect_ratio = native_width / native_height

        # Determine the new dimensions based on the target aspect ratio
        if (original_width / original_height) > target_aspect_ratio:
            # The image is wider than the target aspect ratio
            new_height = original_height
            new_width = int(new_height * target_aspect_ratio)
        else:
            # The image is taller than the target aspect ratio
            new_width = original_width
            new_height = int(new_width / target_aspect_ratio)

        if crop_from_center:
            x_center = round((original_width) / 2)
            y_center = round((original_height) / 2)
            x_start = max(0, x_center - (new_width // 2))
            y_start = max(0, y_center - (new_height // 2))
        else:
            x_start = crop_x
            y_start = crop_y

        if x_start+new_width > original_width:
            x_start = original_width - new_width
        if y_start+new_height > original_height:
            y_start = original_height - new_height

        if is_mask:
            print("is_mask:", is_mask)
            print("Shape du masque avant le recadrage:", image.shape)
            print(f"Paramètres de recadrage pour le masque: y_start={y_start}, new_height={new_height}, x_start={x_start}, new_width={new_width}")
            try:
                # cropped_image = image[y_start:y_start + new_height, x_start:x_start + new_width]
                cropped_image = image[:, y_start:y_start + new_height, x_start:x_start + new_width]
                print("Shape du masque après le recadrage:", cropped_image.shape)
            except Exception as e:
                print(f"Erreur lors du recadrage du masque: {e}")
                print("Taille du masque au moment de l'erreur:", image.size()) # Affiche le nombre total d'éléments
                cropped_image = image  # Pour éviter une erreur ultérieure, même si le masque est vide

            return cropped_image
        else:
            # Convertir la couleur HEX en RGB
            preview_color = torch.tensor(self.Hex_to_RGB(preview_mask_color), dtype=torch.uint8, device=image.device)

            # print("is_mask:", is_mask)
            # Create a preview base with the original image dimensions
            preview = image.clone() if len(image.shape) == 4 else image.clone().unsqueeze(0)

            # Create a white overlay image
            # overlay_image = torch.full((1, original_height, original_width, 3), 255, dtype=torch.uint8, device=image.device)
            overlay_image = torch.full((1, original_height, original_width, 3), 255, dtype=torch.uint8, device=image.device)

            if crop_from_center:
                # Horizontal bands (left and right)
                if x_start > 0:  # Left uncropped region
                    overlay_image[:, :, :x_start, :] = preview_color
                if x_start + new_width < original_width:  # Right uncropped region
                    overlay_image[:, :, x_start + new_width:, :] = preview_color

                # Vertical bands (top and bottom)
                if y_start > 0:  # Top uncropped region
                    overlay_image[:, :y_start, x_start:x_start+new_width, :] = preview_color
                if y_start + new_height < original_height:  # Bottom uncropped region
                    overlay_image[:, y_start + new_height:, x_start:x_start+new_width, :] = preview_color
            else:
                # For non-centered crop, we'll assume it's always from the top
                if y_start > 0:  # Top uncropped region
                    overlay_image[:, :y_start, :, :] = preview_color
                elif y_start + new_height < original_height:  # Bottom uncropped region
                    overlay_image[:, y_start + new_height:, :, :] = preview_color

            # Convert to float and normalize to 0-1 range if needed for further processing
            overlay_image_float = overlay_image.float() / 255.0

            # Make sure preview is also a float tensor
            preview_float = preview.float()
            blend_preview = self.blend_images(preview_float, overlay_image_float, 0.7)

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
        
    def blend_images(self, image1: torch.Tensor, image2: torch.Tensor, blend_factor: float):
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

# Dictionary mapping node names to their corresponding classes
NODE_CLASS_MAPPINGS = {
    "FRED_AutoCropImage_Native_Ratio_v5": FRED_AutoCropImage_Native_Ratio_v5
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_AutoCropImage_Native_Ratio_v5": "👑 FRED_AutoCropImage_Native_Ratio_v5"
}