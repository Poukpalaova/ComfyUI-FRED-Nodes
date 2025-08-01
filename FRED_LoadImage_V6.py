import os
import cv2
import numpy as np
import torch
import hashlib
import folder_paths
import node_helpers
import random
import fnmatch
import json
import time
from PIL import Image, ImageOps, ImageSequence

ALLOWED_EXT = ('.jpeg', '.jpg', '.png', '.tiff', '.gif', '.bmp', '.webp')
CACHE_VERSION = 2  # Increment when cache format changes

HELP_MESSAGE = """This node loads and processes images for use in image generation pipelines.

Key features:
1. Supports loading single images or batches from a specified folder
2. Handles various image formats including JPEG, PNG, TIFF, GIF, BMP, and WebP
3. Processes RGBA images, separating the alpha channel as a mask
4. Calculates image quality score, size, and noise levels
5. Provides options for including subdirectories and handling filename extensions
6. Returns processed image tensor, mask, and various metadata
7. Offers seed-based selection for consistent image loading from folders
"""

def get_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as file:
        for chunk in iter(lambda: file.read(4096), b''):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

class FRED_LoadImage_V6:
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
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT", "INT", "INT", "FLOAT", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "IMAGE_SIZE_KB", "WIDTH", "HEIGHT", "QUALITY_SCORE", "TOTAL IMAGES QTY IN FOLDER(S)", "IMAGES QTY IN CURRENT FOLDER", "SNR", "FOLDER_PATH", "FULL_FOLDER_PATH", "filename_text", "help")

    FUNCTION = "load_image"
    CATEGORY = "👑FRED/image"

    def load_image(self, image, mode="no_folder", seed=0, path="", include_subdirectories=False, filename_text_extension="false"):
        image_path = None

        if mode == "no_folder":
            if isinstance(image, str):
                image_path = folder_paths.get_annotated_filepath(image)
                img = node_helpers.pillow(Image.open, image_path)
                filename = os.path.basename(image_path)
                full_folder_path = os.path.dirname(image_path)
            elif isinstance(image, Image.Image):
                img = image
                filename = "direct_image_input"
                full_folder_path = ""
            else:
                raise ValueError("Invalid image input type.")

            if img.mode == 'RGBA':
                rgb_image = img.convert('RGB')
                alpha_channel = img.split()[3]
                alpha_array = np.array(alpha_channel)
                is_inverted = np.mean(alpha_array) > 127
                image = np.array(rgb_image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]
                mask = np.array(alpha_channel).astype(np.float32) / 255.0
                if is_inverted:
                    mask = 1. - mask
                mask = torch.from_numpy(mask)
            else:
                image = np.array(img.convert("RGB")).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]
                mask = 1. - torch.ones((img.size[1], img.size[0]), dtype=torch.float32)

            output_image = image
            output_mask = mask.unsqueeze(0)
            image_size_kb = int(self.calculate_image_size_in_kb(image_path)) if image_path else 0
            width, height = img.size
            filename_text = filename if filename_text_extension == "true" else os.path.splitext(filename)[0]
            quality_score = self.calculate_image_quality_score(image_size_kb, width, height, os.path.splitext(filename)[1])
            noise_level, snr_value = self.calculate_image_noise(output_image)
            
            total_images_in_folder = 1  # Only one image is loaded in "no_folder" mode
            images_in_current_folder = 1
            full_folder_path = os.path.dirname(image_path)

            return (output_image, output_mask, image_size_kb, width, height, quality_score, total_images_in_folder, images_in_current_folder, snr_value, os.path.dirname(image_path) if image_path else "", full_folder_path, filename_text, HELP_MESSAGE)

        else:
            if not path:
                return self.return_default_image()
            if not os.path.exists(path):
                return self.return_default_image()

            fl = self.BatchImageLoader(path, include_subdirectories)
            max_value = fl.get_total_image_count() # Use the new method to get the count
            if max_value == 0:
                return self.return_default_image()

            image_path = fl.get_image_path_by_id(seed)  # Get the path directly

            if not image_path:
                print("No valid image found, returning default image.")
                return self.return_default_image()
            
            try:
                img = Image.open(image_path)
                img = ImageOps.exif_transpose(img)
                filename = os.path.basename(image_path)
            except (OSError, IOError):
                print(f"Skipping invalid image: {image_path}")
                return self.return_default_image()

            full_folder_path = os.path.dirname(image_path)
            current_folder_path = os.path.dirname(image_path)

            output_images = []
            output_masks = []
            w, h = 0, 0
            excluded_formats = ['MPO']

            for i in ImageSequence.Iterator(img):
                i = node_helpers.pillow(ImageOps.exif_transpose, i)
                if i.mode == 'RGBA':
                    rgb_image = i.convert("RGB")
                    alpha_channel = i.split()[3]
                    image = np.array(rgb_image).astype(np.float32) / 255.0
                    mask = np.array(alpha_channel).astype(np.float32) / 255.0
                elif i.mode == 'I':
                    i = i.point(lambda i: i * (1 / 255))
                    image = np.array(i.convert("RGB")).astype(np.float32) / 255.0
                    mask = 1. - np.ones((i.size[1], i.size[0]), dtype=np.float32)
                else:
                    image = np.array(img.convert("RGB")).astype(np.float32) / 255.0
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

            image_size_kb = int(self.calculate_image_size_in_kb(image_path))
            _, height, width, _ = output_image.shape
            filename_text = filename if filename_text_extension == "true" else os.path.splitext(filename)[0]
            quality_score = self.calculate_image_quality_score(image_size_kb, width, height, os.path.splitext(filename)[1])
            noise_level, snr_value = self.calculate_image_noise(output_image)
            
            images_in_current_folder = self.count_images_in_current_folder(current_folder_path)
            
            total_images_in_folder = fl.get_total_image_count() # Use the cached total count


            return (output_image, output_mask, image_size_kb, width, height, quality_score, total_images_in_folder, images_in_current_folder, snr_value, path, full_folder_path, filename_text, HELP_MESSAGE)

    def return_default_image(self):
        default_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        default_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
        return (default_image, default_mask, 0, 64, 64, 0, 0, 0.0, "", "", "default_image", HELP_MESSAGE)

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
        file_size_in_kb = file_size_in_bytes / 1024 # Convert bytes to kilobytes
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

    def count_images_in_current_folder(self, path):
        count = 0
        for filename in os.listdir(path):
            if filename.lower().endswith(ALLOWED_EXT):
                full_path = os.path.join(path, filename)
                if os.path.isfile(full_path):
                    count += 1
        return count

    class BatchImageLoader:
        def __init__(self, directory_path, include_subdirectories=False):
            self.directory_path = directory_path
            self.include_subdirectories = include_subdirectories
            self.image_paths = []
            self.cache_file = self._get_cache_path()
            self.dir_hash = self._calculate_directory_hash()
            # Try to load from cache
            if not self._load_from_cache():
                self._build_index()
                self._save_to_cache()

        def _get_cache_path(self):
            # Use system temp directory, or wherever you want
            temp_dir = os.path.join(os.path.expanduser("~"), ".fred_image_cache")
            os.makedirs(temp_dir, exist_ok=True)
            safe_name = hashlib.md5((self.directory_path + str(self.include_subdirectories)).encode()).hexdigest()
            return os.path.join(temp_dir, f"{safe_name}.json")

        def _calculate_directory_hash(self):
            # Only hash the path and subdir flag for simplicity
            hash_str = self.directory_path + str(self.include_subdirectories)
            return hashlib.md5(hash_str.encode()).hexdigest()

        def _load_from_cache(self):
            if os.path.exists(self.cache_file):
                try:
                    with open(self.cache_file, 'r') as f:
                        data = json.load(f)
                        if data.get('dir_hash') == self.dir_hash:
                            self.image_paths = data['image_paths']
                            return True
                except Exception as e:
                    print(f"Cache load failed: {e}")
            return False

        def _save_to_cache(self):
            data = {
                'dir_hash': self.dir_hash,
                'image_paths': self.image_paths,
                'timestamp': time.time()
            }
            try:
                with open(self.cache_file, 'w') as f:
                    json.dump(data, f)
            except Exception as e:
                print(f"Cache save failed: {e}")

        def _build_index(self):
            self.image_paths = []
            if self.include_subdirectories:
                for root, _, files in os.walk(self.directory_path):
                    for file in files:
                        if file.lower().endswith(ALLOWED_EXT):
                            self.image_paths.append(os.path.abspath(os.path.join(root, file)))
            else:
                for file in os.listdir(self.directory_path):
                    if file.lower().endswith(ALLOWED_EXT):
                        self.image_paths.append(os.path.abspath(os.path.join(self.directory_path, file)))
            self.image_paths.sort()  # For deterministic order

        def get_total_image_count(self):
            return len(self.image_paths)

        def get_image_path_by_id(self, image_id):
            if not self.image_paths:
                return None
            return self.image_paths[image_id % len(self.image_paths)]

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True

NODE_CLASS_MAPPINGS = {
    "FRED_LoadImage_V6": FRED_LoadImage_V6
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_LoadImage_V6": "👑 FRED Load Image V6"
}