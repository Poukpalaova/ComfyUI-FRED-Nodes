import cv2
from PIL import Image
import numpy as np
import torch
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
# import torchvision.transforms.functional as F
import comfy.utils
from comfy_extras.nodes_mask import ImageCompositeMasked
# from .imagefunc import Hex_to_RGB

# SDXL aspect ratios
ASPECT_RATIOS_SDXL = [
    {"name": "SDXL - 1:1 square 1024x1024", "width": 1024, "height": 1024},
    {"name": "SDXL - 3:4 portrait 896x1152", "width": 896, "height": 1152},
    {"name": "SDXL - 5:8 portrait 832x1216", "width": 832, "height": 1216},
    {"name": "SDXL - 9:16 portrait 768x1344", "width": 768, "height": 1344},
    {"name": "SDXL - 9:21 portrait 640x1536", "width": 640, "height": 1536},
    {"name": "SDXL - 4:3 landscape 1152x896", "width": 1152, "height": 896},
    {"name": "SDXL - 3:2 landscape 1216x832", "width": 1216, "height": 832},
    {"name": "SDXL - 16:9 landscape 1344x768", "width": 1344, "height": 768},
    {"name": "SDXL - 21:9 landscape 1536x640", "width": 1536, "height": 640}
]

# Define a help message
HELP_MESSAGE = """This node automatically crops and resizes images to fit SDXL aspect ratios.

Key features:
1. Auto-find SDXL resolution: Set to True to automatically find the closest SDXL ratio for your image.
2. Custom aspect ratios: Choose from predefined SDXL ratios or set a custom width and height.
3. Cropping options: 
   - Crop from center or adjust using crop_x_in_Percent and crop_y_in_Percent.
   - Option to pre-crop based on an input mask.
4. Resizing:
   - Option to resize the cropped image to the target SDXL dimensions.
   - Different interpolation modes for upscaling and downscaling.
5. Prescaling: Apply a prescale factor to increase or decrease the final image size.
6. Preview: Generates a preview image showing the cropped area.
7. Mask handling: Can process and modify input masks alongside the image.

Use 'Auto_find_SDXL_resolution' for automatic ratio selection, or choose a specific ratio. 
Adjust cropping, resizing, and scaling options to fine-tune the output. 
The node provides both the processed image and a visual preview of the changes."""

class FRED_AutoCropImage_SDXL_Ratio_v4:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "Precrop_from_input_mask": ("BOOLEAN", {"default": False},),
                "aspect_ratio": (["custom"] + ["Auto_find_SDXL_resolution"] + [aspect_ratio["name"] for aspect_ratio in ASPECT_RATIOS_SDXL] + ["no_crop_to_ratio"],),
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
        "SDXL_width",
        "SDXL_height",
        "sd_aspect_ratios",
        "help"
    )
    FUNCTION = "run"
    CATEGORY = "👑FRED/image/postprocessing"
    OUTPUT_NODE = True

    def run(self, image, Precrop_from_input_mask, aspect_ratio, custom_width, custom_height,
            crop_from_center, crop_x_in_Percent, crop_y_in_Percent, resize_image,
            resize_mode_if_upscale, resize_mode_if_downscale, prescale_factor,
            include_prescale_if_resize, preview_mask_color, mask_optional=None):
        
        _, original_height, original_width, _ = image.shape
        modified_image: torch.Tensor
        sd_aspect_ratios = None

        if mask_optional is None:
            mask = torch.zeros(1, original_height, original_width, dtype=torch.float32)
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
            sdxl_width = original_width
            sdxl_height = original_height
        else:
            if aspect_ratio == "Auto_find_SDXL_resolution":
                sdxl_width, sdxl_height, sd_aspect_ratios = self.find_closest_sd_resolution(original_width, original_height)
            elif aspect_ratio == "custom":
                sdxl_width = custom_width
                sdxl_height = custom_height
            else:
                sdxl_width, sdxl_height = [(a["width"], a["height"]) for a in ASPECT_RATIOS_SDXL
                                            if a["name"] == aspect_ratio][0]

            if sdxl_width != original_width and sdxl_height != original_height:
                cropped_image, preview = self.crop_image_to_ratio(image, sdxl_width, sdxl_height, crop_from_center, crop_x_in_Percent, crop_y_in_Percent, False, preview_mask_color)
                cropped_mask = self.crop_image_to_ratio(mask, sdxl_width, sdxl_height, crop_from_center, crop_x_in_Percent, crop_y_in_Percent, True, preview_mask_color)
            else:
                preview = image
                cropped_image = image
                cropped_mask = mask
                sdxl_width = original_width
                sdxl_height = original_height

        _, cropped_height, cropped_width, _ = cropped_image.shape

        if resize_image:
            crop_from_center_str = "center" if crop_from_center else "disabled"
            if include_prescale_if_resize:
                sdxl_width_wfactor = int(sdxl_width * prescale_factor)
                sdxl_height_wfactor = int(sdxl_height * prescale_factor)
                resize_interpolation_mode = resize_mode_if_downscale if sdxl_width_wfactor < cropped_width else resize_mode_if_upscale
                scale_factor = prescale_factor
                resized_image = self.resize_image(cropped_image, resize_interpolation_mode, sdxl_width_wfactor, sdxl_height_wfactor, crop_from_center_str)
                # resized_mask = torch.nn.functional.interpolate(cropped_mask.unsqueeze(0), size=(sdxl_height_wfactor, sdxl_width_wfactor), mode="nearest").squeeze(0).clamp(0.0, 1.0)
                resized_mask = comfy.utils.common_upscale(cropped_mask.unsqueeze(1), sdxl_width_wfactor, sdxl_height_wfactor, resize_interpolation_mode, crop_from_center_str).squeeze(1)
            else:
                resize_interpolation_mode = resize_mode_if_downscale if sdxl_width < cropped_width else resize_mode_if_upscale
                scale_factor = prescale_factor
                resized_image = self.resize_image(cropped_image, resize_interpolation_mode, sdxl_width, sdxl_height, crop_from_center_str)
                # resized_mask = torch.nn.functional.interpolate(cropped_mask.unsqueeze(0), size=(sdxl_height, sdxl_width), mode="nearest").squeeze(0).clamp(0.0, 1.0)
                resized_mask = comfy.utils.common_upscale(cropped_mask.unsqueeze(1), sdxl_width, sdxl_height, resize_interpolation_mode, crop_from_center_str).squeeze(1)
            modified_image = resized_image
            modified_mask = resized_mask
        else:
            scale_factor = prescale_factor * min(sdxl_width / cropped_width, sdxl_height / cropped_height)
            modified_image = cropped_image
            modified_mask = cropped_mask

        if sd_aspect_ratios is None:
            sd_aspect_ratios = aspect_ratio

        _, output_height, output_width, _ = modified_image.shape

        return (
            modified_image,
            preview,
            modified_mask,
            scale_factor,
            output_width,
            output_height,
            sdxl_width,
            sdxl_height,
            sd_aspect_ratios,
            HELP_MESSAGE
        )

    def find_mask_boundaries(self, mask):
        if mask is None:
            return None, None, None, None
        mask_np = mask.squeeze().cpu().numpy()
        rows = np.any(mask_np, axis=1)
        cols = np.any(mask_np, axis=0)
        if not np.any(rows) or not np.any(cols):
            # Le masque est entièrement noir, retourner les dimensions complètes de l'image
            return 0, 0, mask_np.shape[1] - 1, mask_np.shape[0] - 1
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        return x_min, y_min, x_max, y_max

    def resize_image(self, cropped_image, resize_interpolation_mode, width, height, crop_from_center):
        samples = cropped_image.movedim(-1,1)
        resized_image = comfy.utils.common_upscale(samples, width, height, resize_interpolation_mode, crop_from_center)
        resized_image = resized_image.movedim(1,-1)
        return resized_image

    def find_closest_sd_resolution(self, original_width, original_height):
        sd_aspect_ratios = [(a["name"], a["width"], a["height"]) for a in ASPECT_RATIOS_SDXL]
        original_aspect_ratio = original_width / original_height
        closest_distance = float('inf')
        sdxl_width = sdxl_height = 0

        for name, sdxl_width, sdxl_height in sd_aspect_ratios:
            sd_aspect_ratio = sdxl_width / sdxl_height
            ratio_distance = abs(original_aspect_ratio - sd_aspect_ratio)
            dimension_distance = (abs(original_width - sdxl_width) + abs(original_height - sdxl_height)) / (original_width + original_height)
            ratio_weight = 0.5
            dimension_weight = 0.5
            distance = (ratio_weight * ratio_distance) + (dimension_weight * dimension_distance)

            if distance < closest_distance:
                closest_distance = distance
                found_sdxl_width = sdxl_width
                found_sdxl_height = sdxl_height
                found_sd_aspect_ratios = f"{name} - ({sdxl_width}x{sdxl_height})"

        return found_sdxl_width, found_sdxl_height, found_sd_aspect_ratios

    def crop_image_to_ratio(self, image, sdxl_width, sdxl_height, crop_from_center, crop_x_in_Percent, crop_y_in_Percent, is_mask, preview_mask_color):
        if len(image.shape) == 4:
            _, original_height, original_width, _ = image.shape
        elif len(image.shape) == 3:
            _, original_height, original_width = image.shape
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")

        crop_x = round((crop_x_in_Percent * original_width) / 100)
        crop_y = round((crop_y_in_Percent * original_height) / 100)

        # Calculate the target aspect ratio
        target_aspect_ratio = sdxl_width / sdxl_height

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

            # print("overlay_image_float shape:", overlay_image_float.shape)
            # print("preview_float shape:", preview_float.shape)

            # blend_preview = self.blend_images(preview, overlay_mask_image, 0.7)
            blend_preview = self.blend_images(preview_float, overlay_image_float, 0.7)

            # Convertir en numpy array AVANT d'utiliser ascontiguousarray
            blend_preview_np = (blend_preview[0].cpu().numpy() * 255).astype(np.uint8)

            # print("blended preview shape:", blend_preview.shape)
            # blend_preview_np = (blend_preview[0].cpu().numpy() * 255).astype(np.uint8)
            blend_preview_np = np.ascontiguousarray(blend_preview_np)
            # print("preview_color:", preview_color.tolist())
            # print("preview_color:", (int(preview_color[0]), int(preview_color[1]), int(preview_color[2])))
            
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
    "FRED_AutoCropImage_SDXL_Ratio_V4": FRED_AutoCropImage_SDXL_Ratio_v4
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_AutoCropImage_SDXL_Ratio_V4": "👑 FRED_AutoCropImage_SDXL_Ratio_v4"
}
