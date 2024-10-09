import os  # For file path operations
import glob  # For pattern matching of file names
import numpy as np
from PIL import Image, ImageOps  # For image loading and handling EXIF metadata
import torch  # For tensor operations (optional, depending on the function of `pil2tensor`)
from torchvision import transforms  # For tensor conversions from PIL images
from nodes import PreviewImage, SaveImage

# Assuming ALLOWED_EXT is defined somewhere in your codebase
ALLOWED_EXT = ('.jpeg', '.jpg', '.png', '.tiff', '.gif', '.bmp', '.webp')  # Allowed image extensions

class FRED_LoadPathImagesPreview(PreviewImage):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": '', "multiline": False}),
                "pattern": ("STRING", {"default": '*', "multiline": False}),
            },
        }

    NAME = "Images_Preview"
    FUNCTION = "preview_images"
    CATEGORY = "FRED/image"

    class BatchImageLoader:
        def __init__(self, directory_path, pattern):
            self.image_paths = []
            self.load_images(directory_path, pattern)
            self.image_paths.sort()

        def load_images(self, directory_path, pattern):
            for file_name in glob.glob(os.path.join(glob.escape(directory_path), pattern), recursive=True):
                if file_name.lower().endswith(ALLOWED_EXT):
                    abs_file_path = os.path.abspath(file_name)
                    self.image_paths.append(abs_file_path)

    def get_image_by_path(self, image_path):
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image)
        return image

    def preview_images(self, path, pattern='*', filename_prefix="sanmin.preview.", prompt=None, extra_pnginfo=None):
        fl = self.BatchImageLoader(path, pattern)
        images = []
        for image_path in fl.image_paths:
            image = Image.open(image_path)
            tensor_image = pil2tensor(image)
            image = tensor_image[0]
            images.append(image)
        if not images:
            raise ValueError("No images found in the specified path")

        return self.save_images(images, filename_prefix, prompt, extra_pnginfo)

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# Dictionary mapping node names to their corresponding classes
NODE_CLASS_MAPPINGS = {
    "FRED_LoadPathImagesPreview": FRED_LoadPathImagesPreview
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_LoadPathImagesPreview": "👑 FRED_LoadPathImagesPreview"
}