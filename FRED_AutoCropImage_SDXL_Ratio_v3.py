# FRED_AutoCropImage_SDXL_Face_Detect_v1.py is a custom node for ComfyUI
# that which automatically crops an image to fit the SDXL aspect ratio.

from PIL import Image
import numpy as np
import torch
# from torchvision.transforms import Resize
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F
import comfy.utils

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
HELP_MESSAGE = """Put Auto_find_SDXL_resolution to True if you want
the system to find the closest SDXL ratio that fit in your picture. 
If you put it to off, choose a ratio or use Custom to put your custom crop value. 

The image can be resized to the SDXL selected or find ratio with a mode of your choice.
 
If you put a prescale_factor, it will multiply by the scale_factor

If you want to crop from the center, set crop_from_center to True
otherwise, you can adjust crop_x_in_Percent and crop_y_in_Percent to change the cropping area
starting from the top left corner."""

class FRED_AutoCropImage_SDXL_Ratio_v3:
    '''
    Custom node for ComfyUI that which automatically 
    crops an image to fit the SDXL aspect ratio.'''
    
    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "image": ("IMAGE",),
                "Auto_find_SDXL_resolution": ("BOOLEAN", {"default": True},),
                "aspect_ratio": (["custom"] + [aspect_ratio["name"] for aspect_ratio in ASPECT_RATIOS_SDXL] + ["no_crop"],),
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "crop_from_center": ("BOOLEAN", {"default": True},),
                "crop_x_in_Percent": ("INT", {"default": 0, "min": 0, "max": 100}),
                "crop_y_in_Percent": ("INT", {"default": 0, "min": 0, "max": 100}),
                "resize_cropped_image": ("BOOLEAN", {"default": False},),
                "resize_mode_if_upscale": (["bicubic", "bilinear", "nearest", "nearest-exact", "area"], {"default": "bilinear"}),
                "resize_mode_if_downscale": (["bicubic", "bilinear", "nearest", "nearest-exact", "area"], {"default": "area"}),
                "prescale_factor": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 8.0, "step": 0.1}),
                "include_prescale_if_resize": ("BOOLEAN", {"default": False},),
            },
            "optional": {
                "mask_optional": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "FLOAT", "INT", "INT", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = (
        "modified_image", 
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
    CATEGORY = "FRED/image/postprocessing"
    OUTPUT_NODE = True

    def run(self, 
        image, 
        Auto_find_SDXL_resolution, 
        width, 
        height, 
        aspect_ratio, 
        crop_from_center, 
        crop_x_in_Percent, 
        crop_y_in_Percent, 
        resize_cropped_image,
        resize_mode_if_upscale,
        resize_mode_if_downscale,
        prescale_factor,
        include_prescale_if_resize, 
        mask_optional=None
    ):

        _, original_height, original_width, _ = image.shape

        modified_image: torch.Tensor
        # modified_mask: torch.Tensor

        sd_aspect_ratios = None

        # define if mask exist if not it create it or squeeze it
        if mask_optional is None:
            mask = torch.zeros(1, original_height, original_width, dtype=torch.float32)
        else:
            mask = mask_optional
            if mask.shape[1] != original_height or mask.shape[2] != original_width:
                mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=(original_height, original_width), mode="bicubic").squeeze(0).clamp(0.0, 1.0)

        if aspect_ratio == "no_crop":
            cropped_image = image
            cropped_mask = mask
        else:
            if Auto_find_SDXL_resolution:
                sdxl_width, sdxl_height, sd_aspect_ratios = self.find_closest_sd_resolution(original_width, original_height)
            elif aspect_ratio == "custom":
                sdxl_width = width
                sdxl_height = height
            else:
                sdxl_width, sdxl_height = [(a["width"], a["height"]) for a in ASPECT_RATIOS_SDXL 
                                       if a["name"] == aspect_ratio][0]
            cropped_image = self.crop_image_to_ratio(image, sdxl_width, sdxl_height, crop_from_center, crop_x_in_Percent, crop_y_in_Percent)
            cropped_mask = self.crop_image_to_ratio(mask, sdxl_width, sdxl_height, crop_from_center, crop_x_in_Percent, crop_y_in_Percent)
            # Utilisez les dimensions H et W

        _, cropped_height, cropped_width, _ = cropped_image.shape

        if resize_cropped_image:
            if crop_from_center:
                crop_from_center = "center"
            else:
                crop_from_center = "disabled"

            if include_prescale_if_resize:
                sdxl_width_wfactor = int(sdxl_width * prescale_factor)
                sdxl_height_wfactor = int(sdxl_height * prescale_factor)
                if sdxl_width_wfactor < cropped_width:
                    resize_interpolation_mode = resize_mode_if_downscale
                else:
                    resize_interpolation_mode = resize_mode_if_upscale
                scale_factor = 1
                resized_image = self.resize_image(cropped_image, resize_interpolation_mode, sdxl_width_wfactor, sdxl_height_wfactor, crop_from_center)
                # resized_mask = self.resize_image(cropped_mask, resize_interpolation_mode, sdxl_width_wfactor, sdxl_height_wfactor, crop_from_center)
                resized_mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=(original_height, original_width), mode="nearest").squeeze(0).clamp(0.0, 1.0)
            else:
                if sdxl_width < cropped_width:
                    resize_interpolation_mode = resize_mode_if_downscale
                else:
                    resize_interpolation_mode = resize_mode_if_upscale
                resized_image = self.resize_image(cropped_image, resize_interpolation_mode, sdxl_width, sdxl_height, crop_from_center)
                resized_mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=(original_height, original_width), mode="nearest").squeeze(0).clamp(0.0, 1.0)
                scale_factor = prescale_factor
            modified_image = resized_image
            modified_mask = resized_mask
        else:
            # Required scale factor to bring the cropped image to a valid sd scale * user prescale_factor
            scale_factor = prescale_factor * min(sdxl_width / cropped_width, sdxl_height / cropped_height)
            modified_image = cropped_image
            modified_mask = cropped_mask

        if sd_aspect_ratios is None:
            sd_aspect_ratios = aspect_ratio

        _, output_height, output_width, _ = modified_image.shape

        return (
            modified_image, 
            modified_mask, 
            scale_factor, 
            output_width, 
            output_height, 
            sdxl_width, 
            sdxl_height, 
            sd_aspect_ratios, 
            HELP_MESSAGE
        )

    def resize_image(self, cropped_image, resize_interpolation_mode, width, height, crop_from_center):
        # crop_from_center = "center"
        samples = cropped_image.movedim(-1,1)
        resized_image = comfy.utils.common_upscale(samples, width, height, resize_interpolation_mode, crop_from_center)
        resized_image = resized_image.movedim(1,-1)
        return resized_image

    def find_closest_sd_resolution(self, original_width, original_height):
        # Define valid stable diffusion aspect ratio list
        sd_aspect_ratios = [(a["name"], a["width"], a["height"]) for a in ASPECT_RATIOS_SDXL]
        # Calculate the original aspect ratio
        original_aspect_ratio = original_width / original_height

        # Initialize variables to store the closest size
        closest_distance = float('inf')
        sdxl_width = sdxl_height = 0

        # Iterate through the list of aspect ratios to find the closest one
        for name, sdxl_width, sdxl_height in sd_aspect_ratios:
            # Calculate the aspect ratio
            sd_aspect_ratio = sdxl_width / sdxl_height
            
            # Calculer la différence de ratio comme auparavant
            ratio_distance = abs(original_aspect_ratio - sd_aspect_ratio)

            # Introduire une mesure de distance basée sur les dimensions
            dimension_distance = (abs(original_width - sdxl_width) + abs(original_height - sdxl_height)) / (original_width + original_height)

            # Combinez les distances en utilisant une pondération pour équilibrer leur importance
            # Vous pouvez ajuster les poids (par exemple, ratio_weight et dimension_weight) selon vos besoins
            ratio_weight = 0.5
            dimension_weight = 0.5
            distance = (ratio_weight * ratio_distance) + (dimension_weight * dimension_distance)

            if distance < closest_distance:
                closest_distance = distance
                found_sdxl_width = sdxl_width
                found_sdxl_height = sdxl_height
                found_sd_aspect_ratios = f"{name} - ({sdxl_width}x{sdxl_height})"
        
        return found_sdxl_width, found_sdxl_height, found_sd_aspect_ratios

    def crop_image_to_ratio(self, image, sdxl_width, sdxl_height, crop_from_center, crop_x_in_Percent, crop_y_in_Percent):
        if len(image.shape) == 4:
            _, original_height, original_width, _ = image.shape
        elif len(image.shape) == 3:
            _, original_height, original_width = image.shape
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
        # _, original_height, original_width, _ = image.shape   # 2160x3840
        crop_x = round((crop_x_in_Percent * original_width) / 100)
        crop_y = round((crop_y_in_Percent * original_height) / 100)

        # Calculate the target aspect ratio
        target_aspect_ratio = sdxl_width / sdxl_height # 512/768 = 0.6666666

        # Determine the new dimensions based on the target aspect ratio
        if (original_width / original_height) > target_aspect_ratio:  # 1.777777 > 0.666666
            # The image is wider than the target aspect ratio
            new_height = original_height   # 2160
            new_width = int(new_height * target_aspect_ratio) # 2160*0.666666 = 1440
        else:
            # The image is taller than the target aspect ratio
            new_width = original_width
            new_height = int(new_width / target_aspect_ratio)

        if crop_from_center:
            x_center = round((original_width) / 2) # 1920
            y_center = round((original_height) / 2) # 1080
            x_start = max(0, x_center - (new_width // 2)) #  1920-(1440/2)=1200
            y_start = max(0, y_center - (new_height // 2)) # 1080-(2160/2)=0
        else:
            x_start = crop_x
            y_start = crop_y
            if x_start+new_width > original_width:
                x_start = original_width - new_width
            if y_start+new_height > original_height:
                y_start = original_height - new_height

        # Crop the image
        if len(image.shape) == 4:
            cropped_image = image[:, y_start:y_start+new_height, x_start:x_start+new_width, :] # [:, 0:2160, 1200:2640, :]
        elif len(image.shape) == 3:
            cropped_image = image[:, y_start:y_start+new_height, x_start:x_start+new_width] # [:, 0:2160, 1200:2640, :]
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")

        return cropped_image

# Dictionary mapping node names to their corresponding classes
NODE_CLASS_MAPPINGS = {
    "FRED_AutoCropImage_SDXL_Ratio_V3": FRED_AutoCropImage_SDXL_Ratio_v3
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_AutoCropImage_SDXL_Ratio_V3": "👑 FRED_AutoCropImage_SDXL_Ratio_v3"
}
