import os
import cv2
import numpy as np
import torch
import hashlib
import folder_paths
import node_helpers
import random
import fnmatch
from PIL import Image, ImageOps, ImageSequence

ALLOWED_EXT = ('.jpeg', '.jpg', '.png', '.tiff', '.gif', '.bmp', '.webp')

# Define a help message
HELP_MESSAGE = """This node loads and processes images for use in image generation pipelines.

Key features:

1. Supports loading single images or batches from a specified folder
2. Handles various image formats including JPEG, PNG, TIFF, GIF, BMP, and WebP
3. Processes RGBA images, separating the alpha channel as a mask
4. Calculates image quality score, size, and noise levels
5. Provides options for including subdirectories and handling filename extensions
6. Returns processed image tensor, mask, and various metadata
7. Offers seed-based selection for consistent image loading from folders"""

def get_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as file:
        for chunk in iter(lambda: file.read(4096), b''):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

class FRED_LoadImage_V5:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(ALLOWED_EXT)]
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
                "mode": (["no_folder", "image_from_folder"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 15000, "step": 1}),
                "path": ("STRING", {"default": '', "multiline": False}),
                "include_subdirectories": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "filename_text_extension": (["true", "false"], {"default": "false"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT", "INT", "FLOAT", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "IMAGE_SIZE_KB", "WIDTH", "HEIGHT", "QUALITY_SCORE", "IMAGES QUANTITY IN FOLDER", "SNR", "FOLDER_PATH", "filename_text", "help")
    FUNCTION = "load_image"
    CATEGORY = "FRED/image"

    # def load_image(self, seed, image, mask=None, mode="no_folder", path="", filename_text_extension="false"):
    def load_image(self, image, mode="no_folder", seed=0, path="", include_subdirectories=False, filename_text_extension="false"):
        image_path = None
        # print("folder path:", path)

        if mode == "no_folder":
            if isinstance(image, str):
                image_path = folder_paths.get_annotated_filepath(image)
                img = node_helpers.pillow(Image.open, image_path)
                filename = os.path.basename(image_path)
            elif isinstance(image, Image.Image):
                img = image
                filename = "direct_image_input"
            else:
                raise ValueError("Invalid image input type.")
            print("img.mode:", img.mode)
            if img.mode == 'RGBA':
                # Split the RGBA image into RGB and alpha
                rgb_image = img.convert('RGB')
                alpha_channel = img.split()[3]
                
                # Vérifier si le masque est principalement blanc ou noir
                alpha_array = np.array(alpha_channel)
                is_inverted = np.mean(alpha_array) > 127
    
                # Process RGB image
                image = np.array(rgb_image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]
                
                # Process alpha channel as mask
                mask = np.array(alpha_channel).astype(np.float32) / 255.0
                # mask = 1. - torch.from_numpy(mask)
                # mask = torch.from_numpy(mask)
                if is_inverted:
                    # Si le masque est principalement blanc, l'inverser
                    mask = 1. - mask
    
                mask = torch.from_numpy(mask)
            else:
                # Process non-RGBA images as before
                image = np.array(img.convert("RGB")).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]
                mask = 1. - torch.ones((img.size[1], img.size[0]), dtype=torch.float32)
                # mask = np.array(img.getchannel('A')).astype(np.float32) / 255.0
                # mask = 1. - torch.from_numpy(mask)

            output_image = image
            output_mask = mask.unsqueeze(0)

        else:
            if not path:
                print("No folder path provided, returning default image.")
                return self.return_default_image()
            if not os.path.exists(path):
                print(f"The path '{path}' does not exist. Returning default image.")
                return self.return_default_image()

            fl = self.BatchImageLoader(path, include_subdirectories)
            max_value = len(fl.image_paths)
            if max_value == 0:
                print("No images found in the specified folder.")
                return self.return_default_image()

            # seed = seed % max_value
            # print("seed is:", seed)
            img, filename = fl.get_image_by_id(seed)
            if img is None:
                print("No valid image found, returning default image.")
                return self.return_default_image()

            image_path = fl.image_paths[seed]

            # Process the image
            output_images = []
            output_masks = []
            w, h = 0, 0
            excluded_formats = ['MPO']

            for i in ImageSequence.Iterator(img):
                i = node_helpers.pillow(ImageOps.exif_transpose, i)
                
                if i.mode == 'RGBA':
                    # Handle RGBA images
                    rgb_image = i.convert("RGB")
                    alpha_channel = i.split()[3]
                    
                    image = np.array(rgb_image).astype(np.float32) / 255.0
                    mask = np.array(alpha_channel).astype(np.float32) / 255.0
                    # mask = 1. - mask  # Invert the mask
                elif i.mode == 'I':
                    i = i.point(lambda i: i * (1 / 255))
                    image = np.array(i.convert("RGB")).astype(np.float32) / 255.0
                    # mask = np.ones((i.size[1], i.size[0]), dtype=np.float32)
                    mask = 1. - np.ones((i.size[1], i.size[0]), dtype=np.float32)
                else:
                    image = np.array(i.convert("RGB")).astype(np.float32) / 255.0
                    # mask = np.ones((i.size[1], i.size[0]), dtype=np.float32)
                    mask = 1. - np.ones((i.size[1], i.size[0]), dtype=np.float32)
                
                if len(output_images) == 0:
                    w, h = image.shape[1], image.shape[0]
                elif image.shape[1] != w or image.shape[0] != h:
                    continue
                
                image = torch.from_numpy(image)[None,]
                mask = torch.from_numpy(mask)
                
                output_images.append(image)
                output_masks.append(mask.unsqueeze(0))

            if len(output_images) > 1 and img.format not in excluded_formats:
                output_image = torch.cat(output_images, dim=0)
                output_mask = torch.cat(output_masks, dim=0)
            else:
                output_image = output_images[0]
                output_mask = output_masks[0]

        # Retrieve image size
        # print("image path:", image_path)
        image_size_kb = int(self.calculate_image_size_in_kb(image_path))

        _, height, width, _ = output_image.shape
        filename_text = filename if filename_text_extension == "true" else os.path.splitext(filename)[0]
        quality_score = self.calculate_image_quality_score(image_size_kb, width, height, os.path.splitext(filename)[1])
        noise_level, snr_value = self.calculate_image_noise(output_image)

        # Check if the provided path is a directory only for folder modes
        if mode != "no_folder" and not os.path.isdir(path):
            raise ValueError(f"The path '{path}' is not a valid directory.")

        # List of valid image extensions
        valid_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.webp']
        images_count = 0

        # Iterate through each file in the directory only for folder modes
        if mode != "no_folder":
            for filename in os.listdir(path):
                if any(fnmatch.fnmatch(filename, ext) for ext in valid_extensions):
                    full_path = os.path.join(path, filename)
                    if os.path.isfile(full_path):
                        images_count += 1

        return (output_image, output_mask, image_size_kb, width, height, quality_score, images_count, snr_value, path, filename_text, HELP_MESSAGE)

    def return_default_image(self):
        # default_image = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
        # default_mask = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
        # return (default_image, default_mask, 0, 64, 64, 0, 0, 0, "", "default_image", 0)
        default_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        default_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
        return (default_image, default_mask, 0, 64, 64, 0, 0, 0.0, "", "default_image")

    def calculate_image_quality_score(self, image_size_kb, width, height, image_format):
        uncompressed_size = (width * height * 3) // 1024
        pixel_size = width * height
        
        if uncompressed_size == 0 or pixel_size == 0:
            return 0
        
        score = min(100, int((image_size_kb / uncompressed_size) * 100))
        
        format_multipliers = {'jpeg': 0.8, 'jpg': 0.8, 'png': 1.1, 'webp': 1.2}
        score *= format_multipliers.get(image_format.lower()[1:], 1.0)
        
        score_adjusted = score * (pixel_size / (width * height))
        return max(0, min(100, score_adjusted))

    def calculate_image_size_in_kb(self, image_path):
        file_size_in_bytes = os.path.getsize(image_path)
        file_size_in_kb = file_size_in_bytes / 1024  # Convert bytes to kilobytes
        return file_size_in_kb

    def calculate_image_noise(self, image_tensor):
        image_np = image_tensor.squeeze().cpu().numpy() * 255.0
        image_np = image_np.astype(np.uint8)
        
        if image_np.shape[0] == 3:
            image_np = np.transpose(image_np, (1, 2, 0))
        
        image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        mean = cv2.blur(image_gray, (3, 3))
        diff = image_gray - mean
        variance = np.mean(diff ** 2)
        
        signal_power = np.mean(image_gray ** 2)
        noise_power = variance
        snr = 10 * np.log10(signal_power / noise_power) if noise_power != 0 else float('inf')
        
        return variance, snr

    class BatchImageLoader:
        def __init__(self, directory_path, include_subdirectories=False):
            self.image_paths = []
            self.load_images(directory_path, include_subdirectories)
            self.image_paths.sort()

        def load_images(self, directory_path, include_subdirectories):
            if include_subdirectories:
                for root, _, files in os.walk(directory_path):
                    for file_name in files:
                        if file_name.lower().endswith(ALLOWED_EXT):
                            abs_file_path = os.path.abspath(os.path.join(root, file_name))
                            self.image_paths.append(abs_file_path)
            else:
                for file_name in os.listdir(directory_path):
                    if file_name.lower().endswith(ALLOWED_EXT):
                        abs_file_path = os.path.abspath(os.path.join(directory_path, file_name))
                        if os.path.isfile(abs_file_path):
                            self.image_paths.append(abs_file_path)

        def get_image_by_id(self, image_id):
            # print("image_id is:", image_id)
            while image_id < len(self.image_paths):
                try:
                    i = Image.open(self.image_paths[image_id])
                    i = ImageOps.exif_transpose(i)
                    return i, os.path.basename(self.image_paths[image_id])
                except (OSError, IOError):
                    image_id += 1
                    print(f"Skipping invalid image at seed `{image_id}`")
            return None, None

        # def get_mask_by_id(self, image_id):
            # if 0 <= image_id < len(self.mask_paths):
                # try:
                    # mask = Image.open(self.mask_paths[image_id])
                    # return mask, os.path.basename(self.mask_paths[image_id])
                # except:
                    # print(f"Skipping invalid mask at seed `{image_id}`")
                    # return None, None

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if kwargs['mode'] == 'no_folder':
            return float("NaN")
        else:
            # fl = FRED_LoadImage_V5.BatchImageLoader(kwargs['path'])
            # filename = fl.get_current_image()
            # image = os.path.join(kwargs['path'], filename)
            # sha = get_sha256(image)
            # return sha
            fl = FRED_LoadImage_V5.BatchImageLoader(kwargs['path'])
            if len(fl.image_paths) > 0:
                image = fl.image_paths[0]  # Prend le premier fichier image
                sha = get_sha256(image)
                return sha
            return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True
NODE_CLASS_MAPPINGS = {
    "FRED_LoadImage_V5": FRED_LoadImage_V5
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_LoadImage_V5": "👑 FRED Load Image V5"
}