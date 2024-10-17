import cv2
from PIL import Image
import numpy as np
import torch
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F
import comfy.utils
from comfy_extras.nodes_mask import ImageCompositeMasked

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

class FRED_AutoCropImage_SDXL_Ratio_v4:
    '''
    Custom node for ComfyUI that which automatically
    crops an image to fit the SDXL aspect ratio.
    '''
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "Precrop_from_input_mask": ("BOOLEAN", {"default": False},),
                "aspect_ratio": (["custom"] + ["Auto_find_SDXL_resolution"] + [aspect_ratio["name"] for aspect_ratio in ASPECT_RATIOS_SDXL] + ["no_crop"],),
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
    CATEGORY = "FRED/image/postprocessing"
    OUTPUT_NODE = True

    def run(self, image, Precrop_from_input_mask, aspect_ratio, custom_width, custom_height,
            crop_from_center, crop_x_in_Percent, crop_y_in_Percent, resize_image,
            resize_mode_if_upscale, resize_mode_if_downscale, prescale_factor,
            include_prescale_if_resize, mask_optional=None):
        
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

        if aspect_ratio == "no_crop":
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
                cropped_image, preview = self.crop_image_to_ratio(image, sdxl_width, sdxl_height, crop_from_center, crop_x_in_Percent, crop_y_in_Percent)
                cropped_mask = self.crop_image_to_ratio(mask, sdxl_width, sdxl_height, crop_from_center, crop_x_in_Percent, crop_y_in_Percent)[0]
            else:
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
                scale_factor = 1
                resized_image = self.resize_image(cropped_image, resize_interpolation_mode, sdxl_width_wfactor, sdxl_height_wfactor, crop_from_center_str)
                resized_mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=(original_height, original_width), mode="nearest").squeeze(0).clamp(0.0, 1.0)
            else:
                resize_interpolation_mode = resize_mode_if_downscale if sdxl_width < cropped_width else resize_mode_if_upscale
                resized_image = self.resize_image(cropped_image, resize_interpolation_mode, sdxl_width, sdxl_height, crop_from_center_str)
                resized_mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=(original_height, original_width), mode="nearest").squeeze(0).clamp(0.0, 1.0)
            scale_factor = prescale_factor
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
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        return x_min, y_min, x_max, y_max

    # def generate_preview(self, original_image, sdxl_width, sdxl_height, crop_from_center, crop_x_in_Percent, crop_y_in_Percent):
        # # Convert the original image to numpy
        # original_np = original_image[0].cpu().numpy().transpose(1, 2, 0)
        # original_np = (original_np * 255).astype(np.uint8)

        # # Get the dimensions of the original image
        # oh, ow = original_np.shape[:2]

        # # Calculate the coordinates of the rectangle
        # if crop_from_center:
            # x = (ow - sdxl_width) // 2
            # y = (oh - sdxl_height) // 2
        # else:
            # x = int(crop_x_in_Percent * ow / 100)
            # y = int(crop_y_in_Percent * oh / 100)

        # # Ensure the coordinates don't exceed the image boundaries
        # x = max(0, min(x, ow - sdxl_width))
        # y = max(0, min(y, oh - sdxl_height))

        # # Create a copy of the original image to draw on
        # preview = original_np.copy()

        # # Create a semi-transparent blue overlay
        # overlay = preview.copy()
        # cv2.rectangle(overlay, (x, y), (x + sdxl_width, y + sdxl_height), (0, 0, 255), -1)

        # # Apply the overlay
        # cv2.addWeighted(overlay, 0.5, preview, 0.5, 0, preview)

        # # Draw the rectangle border
        # cv2.rectangle(preview, (x, y), (x + sdxl_width, y + sdxl_height), (0, 0, 255), 5)

        # # Add text to indicate the crop size
        # cv2.putText(preview, f"Crop: {sdxl_width}x{sdxl_height}", (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # # Convert the preview to a PyTorch tensor
        # preview_tensor = torch.from_numpy(preview.transpose(2, 0, 1)).float() / 255.0

        # return preview

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

    def crop_image_to_ratio(self, image, sdxl_width, sdxl_height, crop_from_center, crop_x_in_Percent, crop_y_in_Percent):
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

        # Create a preview base with the original image dimensions
        preview = image.clone() if len(image.shape) == 4 else image.clone().unsqueeze(0)

        # Create an overlay mask for the uncropped regions
        overlay_mask = torch.zeros((original_height, original_width), dtype=torch.float32, device=image.device)

        if crop_from_center:
            if x_start > 0:  # Left uncropped region
                overlay_mask[:, :x_start] = 1
            if x_start + new_width < original_width:  # Right uncropped region
                overlay_mask[:, x_start + new_width:] = 1
            if y_start > 0:  # Top uncropped region
                overlay_mask[:y_start, :] = 1
            if y_start + new_height < original_height:  # Bottom uncropped region
                overlay_mask[y_start + new_height:, :] = 1
        else:
            if y_start > 0:  # Top uncropped region
                overlay_mask[:y_start, :] = 1
            elif y_start + new_height < original_height:  # Bottom uncropped region
                overlay_mask[y_start + new_height:, :] = 1

        # Create overlay_mask_image with the same format as preview
        overlay_mask_image = 1. - overlay_mask.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, 3)

        # Ensure overlay_mask_image is a float tensor
        overlay_mask_image = overlay_mask_image.float()

        # Make sure preview is also a float tensor
        preview = preview.float()
        
        # Ensure overlay_mask_image has the same shape as preview
        # if preview.shape != overlay_mask_image.shape:
            # overlay_mask_image = overlay_mask_image.expand(preview.shape[0], -1, -1, -1)
        # preview = self.blend_images(preview, overlay_mask_image, 0.7, "multiply")
        # print(f"Type of preview.size: {type(preview.size)}, Value: {preview.size}")
        # size = tuple(preview.size) if not isinstance(preview.size, tuple) else preview.size
        # multiply_color = Image.new("RGBA", size, (128, 128, 128, 255))
        # blended = Image.composite(multiply_color, preview, overlay_mask_image)

        # Draw rectangle for the cropped region
        cv2.rectangle(preview[0].cpu().numpy(), (x_start, y_start), (x_start + new_width, y_start + new_height), (0, 0, 255), 4)

        # Crop the image
        if len(image.shape) == 4:
            cropped_image = image[:, y_start:y_start + new_height, x_start:x_start + new_width, :]
        elif len(image.shape) == 3:
            cropped_image = image[y_start:y_start + new_height, x_start:x_start + new_width]
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")

        return cropped_image, overlay_mask_image
        
    # def blend_images(self, image1: torch.Tensor, image2: torch.Tensor, blend_factor: float, blend_mode: str):
        # image2 = image2.to(image1.device)
        # if image1.shape != image2.shape:
            # image2 = image2.permute(0, 3, 1, 2)
            # image2 = comfy.utils.common_upscale(image2, image1.shape[2], image1.shape[1], upscale_method='bicubic', crop='center')
            # image2 = image2.permute(0, 2, 3, 1)

        # blended_image = self.blend_mode(image1, image2, blend_mode)
        # blended_image = image1 * (1 - blend_factor) + blended_image * blend_factor
        # blended_image = torch.clamp(blended_image, 0, 1)
        # return (blended_image,)
        
    # def blend_mode(self, img1, img2, mode):
        # if mode == "normal":
            # return img2
        # elif mode == "multiply":
            # return img1 * img2
        # elif mode == "screen":
            # return 1 - (1 - img1) * (1 - img2)
        # elif mode == "overlay":
            # return torch.where(img1 <= 0.5, 2 * img1 * img2, 1 - 2 * (1 - img1) * (1 - img2))
        # elif mode == "soft_light":
            # return torch.where(img2 <= 0.5, img1 - (1 - 2 * img2) * img1 * (1 - img1), img1 + (2 * img2 - 1) * (self.g(img1) - img1))
        # elif mode == "difference":
            # return img1 - img2
        # else:
            # raise ValueError(f"Unsupported blend mode: {mode}")

# Dictionary mapping node names to their corresponding classes
NODE_CLASS_MAPPINGS = {
    "FRED_AutoCropImage_SDXL_Ratio_V4": FRED_AutoCropImage_SDXL_Ratio_v4
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_AutoCropImage_SDXL_Ratio_V4": "👑 FRED_AutoCropImage_SDXL_Ratio_v4"
}


