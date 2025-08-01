import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from .utils import tensor2cv, cv2tensor, pil2tensor

class FRED_ImageUncropFromBBox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "cropped_image": ("IMAGE",),
                "bbox": ("BBOX",),
                "border_blending": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "use_square_mask": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "optional_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "uncrop_with_bbox"
    CATEGORY = "👑FRED/image/postprocessing"

    def create_mask(self, width, height, border_blending, use_square_mask):
        """Creates a mask for blending."""
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)

        if use_square_mask:
            draw.rectangle((0, 0, width, height), fill=255)
        else:
            draw.ellipse((0, 0, width, height), fill=255)

        if border_blending > 0:
            blur_radius = int(min(width, height) * border_blending)
            if blur_radius > 0:
                mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))

        return mask

    def uncrop_with_bbox(self, original_image: torch.Tensor, cropped_image: torch.Tensor, bbox, border_blending, use_square_mask, optional_mask: torch.Tensor = None):
        """
        Uncrops the image using the provided bounding box, with optional mask and border blending.
        """

        # Validate bbox format
        if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
            raise ValueError(f"Invalid bounding box format: {bbox}")

        x, y, width, height = map(int, bbox)

        # Ensure original_image and cropped_image have the same data type and device
        if original_image.dtype != cropped_image.dtype:
            cropped_image = cropped_image.to(original_image.dtype)
        if original_image.device != cropped_image.device:
            cropped_image = cropped_image.to(original_image.device)

        # Create a copy of the original image to avoid modifying it directly
        output_image = original_image.clone()

        # Generate mask for blending
        mask_pil = self.create_mask(width, height, border_blending, use_square_mask)
        mask = pil2tensor(mask_pil).unsqueeze(0).float() / 255.0  # Normalize to [0, 1] and add batch dimension

        # Apply optional mask if provided
        if optional_mask is not None:
            # Resize the optional mask to the size of the cropped image
            optional_mask_resized = torch.nn.functional.interpolate(
                optional_mask.unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions
                size=(height, width),
                mode="bilinear",
                align_corners=False
            ).squeeze(0).squeeze(0)  # Remove batch and channel dimensions
            mask = mask * optional_mask_resized

        # Expand mask to have the same number of channels as the images
        mask = mask.unsqueeze(-1).expand(-1, -1, height, width, cropped_image.shape[3])

        # Perform blending
        output_image[:, y:y + height, x:x + width, :] = (
            cropped_image * mask + output_image[:, y:y + height, x:x + width, :] * (1 - mask)
        )
        return (output_image,)
    @staticmethod
    def IS_CHANGED(*args, **kwargs):
        return False

# Dictionary mapping node names to their corresponding classes
NODE_CLASS_MAPPINGS = {
    "FRED_ImageUncropFromBBox": FRED_ImageUncropFromBBox
}

# Dictionary mapping node display names to their corresponding class names
NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_ImageUncropFromBBox": "👑 FRED_ImageUncropFromBBox"
}