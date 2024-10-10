<<<<<<< HEAD
import os
from PIL import Image, ImageOps, ImageSequence
import numpy as np
import torch
import hashlib
import folder_paths
import node_helpers
import cv2
import glob
import random
import fnmatch

ALLOWED_EXT = ('.jpeg', '.jpg', '.png', '.tiff', '.gif', '.bmp', '.webp')

# SHA-256 Hash
def get_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as file:
        for chunk in iter(lambda: file.read(4096), b''):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

class FRED_LoadImage_V3:
    def __init__(self):
        self.current_index = 0  # Initialize the index

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
                "mode": (["no_folder", "single_image_from_folder", "incremental_image_from_folder", "random_from_folder"],),
                "index": ("INT", {"default": 0, "min": 0, "max": 150000, "step": 1}),
                "path": ("STRING", {"default": '', "multiline": False}),
            },
            "optional": {
                "filename_text_extension": (["true", "false"], {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT", "INT", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = (
        "IMAGE", 
        "MASK", 
        "IMAGE_SIZE_KB", 
        "WIDTH", 
        "HEIGHT", 
        "QUALITY_SCORE", 
        "IMAGES QUANTITY IN FOLDER", 
        "SNR", 
        "FOLDER_PATH", 
        "filename_text"
    )
    FUNCTION = "load_image"
    CATEGORY = "FRED/image"
    
    def load_image(self, image, mode="no_folder", index=0, path="", filename_text_extension="false"):
        # Initialize image_path
        image_path = None
        print("folder path:", path)

        if mode == "no_folder":
            # Execute normally for mode "no_folder"
            if isinstance(image, str):
                image_path = folder_paths.get_annotated_filepath(image)
                img = node_helpers.pillow(Image.open, image_path)
                filename = os.path.basename(image_path)
            elif isinstance(image, Image.Image):
                img = image
                filename = "direct_image_input"
            else:
                raise ValueError("Invalid image input type.")
        else:
            # Check if path is empty and handle accordingly
            if not path:
                print("No folder path provided, returning default image.")
                return self.return_default_image()

            if not os.path.exists(path):
                print(f"The path '{path}' does not exist. Returning default image.")
                return self.return_default_image()

            fl = self.BatchImageLoader(path)

            # Handle modes accordingly
            if mode == 'single_image_from_folder':
                img, filename = fl.get_image_by_id(index)
                # index += 1

            elif mode == 'incremental_image_from_folder':
                img, filename = fl.get_image_by_id(self.current_index)
                self.current_index += 1
                if self.current_index >= len(fl.image_paths):
                    self.current_index = 0

            else:  # random_from_folder
                # newindex = random.randint(0, len(fl.image_paths) - 1)
                # img, filename = fl.get_image_by_id(newindex)
                self.current_index = random.randint(0, len(fl.image_paths) - 1)
                img, filename = fl.get_image_by_id(self.current_index)

            if img is None:
                print("No valid image found, returning default image.")
                return self.return_default_image()

            image_path = os.path.join(path, filename)

        # Process the image
        output_images = []
        output_masks = []
        w, h = 0, 0 # Initialize with default values

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]
            
            if image.size[0] != w or image.size[1] != h:
                continue
            
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        # Retrieve image size
        # image_size_kb = int(self.calculate_image_size_in_kb(image_path)) if isinstance(image, str) else 0
        print("image path:", image_path)
        image_size_kb = int(self.calculate_image_size_in_kb(image_path))
        _, height, width, _ = output_image.shape
        # print("is image instance:", isinstance(image, str))

        # Retrieve file name
        if filename_text_extension == "true":
            filename_text = filename
        else:
            filename_text = os.path.splitext(filename)[0]

        # Calculate image quality score
        quality_score = self.calculate_image_quality_score(image_size_kb, width, height)

        noise_level, snr_value = self.calculate_image_noise(image)

        # Check if the provided path is a directory only for folder modes
        if mode != "no_folder" and not os.path.isdir(path):
            raise ValueError(f"The path '{path}' is not a valid directory.")
        
        # List of valid image extensions
        valid_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.webp']

        images_count = 0

        # Iterate through each file in the directory only for folder modes
        if mode != "no_folder":
            # for filename in os.listdir(image_path):
            for filename in os.listdir(path):
                if any(fnmatch.fnmatch(filename, ext) for ext in valid_extensions):
                    # full_path = os.path.join(image_path, filename)
                    full_path = os.path.join(path, filename)
                    if os.path.isfile(full_path):
                        images_count += 1

        return (output_image, output_mask, image_size_kb, width, height, quality_score, images_count, snr_value, path, filename_text)

    def return_default_image(self):
        # Logic to create or load a default image
        default_image = np.zeros((64, 64, 3), dtype=np.float32)  # Example: a blank image
        default_image_tensor = torch.from_numpy(default_image)[None, :]  # Add batch dimension
        default_mask = torch.zeros((1, 64, 64), dtype=torch.float32)  # Example: a blank mask
        image_size_kb = 0  # Default size
        width, height = 64, 64  # Default dimensions
        quality_score = 0  # Default quality score
        images_count = 0  # Default image count
        snr_value = 0  # Default SNR value
        filename_text = "default_image"

        return (default_image_tensor, default_mask, image_size_kb, width, height, quality_score, images_count, snr_value, filename_text)

    def calculate_image_quality_score(self, image_size_kb, width, height):
        # Taille non compressée en Ko
        uncompressed_size = (width * height * 3) / 1024  # Converti à Ko
        uncompressed_size = int(uncompressed_size)  # Assurez-vous que c'est un entier

        # Calculer la taille de l'image en pixels
        pixel_size = width * height

        # Éviter la division par zéro
        if uncompressed_size == 0 or pixel_size == 0:
            return 0

        # Calculer le score basé sur la taille de l'image et la taille du fichier
        if image_size_kb >= uncompressed_size:
            score = 100
        else:
            score = int((image_size_kb / uncompressed_size) * 100)

        # Ajuster le score en fonction de la taille de l'image
        score_adjusted = score * (pixel_size / (width * height))

        return max(0, min(100, score_adjusted))  # Assurez-vous que le score reste entre 0 et 100

    def calculate_image_size_in_kb(self, image_path):
        file_size_in_bytes = os.path.getsize(image_path)
        file_size_in_kb = file_size_in_bytes / 1024  # Convert bytes to kilobytes
        return file_size_in_kb

    def calculate_image_noise(self, image_tensor):
        """
        Calcule le niveau de bruit dans une image à l'aide de la variance locale et du SNR.
        
        Args:
            image_tensor: torch.Tensor, image en format PyTorch.

        Returns:
            noise_level: float, estimation du niveau de bruit par la variance locale.
            snr: float, rapport signal/bruit.
        """
        # Convertir le tenseur PyTorch en tableau NumPy
        image_np = image_tensor.squeeze().cpu().numpy() * 255.0
        image_np = image_np.astype(np.uint8)

        # Vérifier si l'image est dans le bon format (H, W, C)
        if image_np.shape[0] in [1, 3]:
            # Si c'est le cas, permuter pour obtenir (H, W, C)
            image_np = np.transpose(image_np, (1, 2, 0))

        # Si l'image a 3 canaux, on la convertit en niveau de gris
        if image_np.shape[2] == 3:
            image_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("L'image n'a pas un format de canaux valide : {}".format(image_np.shape))

        # Variance locale comme mesure de bruit
        mean = cv2.blur(image_gray, (3, 3))  # Moyenne locale
        diff = image_gray - mean
        variance = np.mean(diff ** 2)  # Variance locale des pixels voisins (bruit)

        # Calcul du SNR (Signal-to-Noise Ratio)
        signal_power = np.mean(image_gray ** 2)  # Puissance du signal (moyenne des carrés des pixels)
        noise_power = variance  # Puissance du bruit estimée par la variance locale
        snr = 10 * np.log10(signal_power / noise_power) if noise_power != 0 else float('inf')
        
        return variance, snr

    class BatchImageLoader:
        def __init__(self, directory_path):
            self.image_paths = []
            self.index = 0
            self.load_images(directory_path)
            self.image_paths.sort()

        def load_images(self, directory_path):
            for root, _, files in os.walk(directory_path):
                for file_name in files:
                    if file_name.lower().endswith(ALLOWED_EXT):
                        abs_file_path = os.path.abspath(os.path.join(root, file_name))
                        self.image_paths.append(abs_file_path)

        def get_image_by_id(self, image_id):
            while image_id < len(self.image_paths):
                try:
                    i = Image.open(self.image_paths[image_id])
                    i = ImageOps.exif_transpose(i)
                    return i, os.path.basename(self.image_paths[image_id])
                except (OSError, IOError):
                    image_id += 1
                    print(f"Skipping invalid image at index `{image_id}`")
            return None, None

        def get_next_image(self):
            while self.index < len(self.image_paths):
                try:
                    image_path = self.image_paths[self.index]
                    self.index += 1
                    if self.index == len(self.image_paths):
                        self.index = 0
                    i = Image.open(image_path)
                    i = ImageOps.exif_transpose(i)
                    return i, os.path.basename(image_path)
                except (OSError, IOError):
                    print(f"Skipping invalid image at index `{self.index}`")
                    self.index += 1
            return None, None

        def get_current_image(self):
            if self.index >= len(self.image_paths):
                self.index = 0
            image_path = self.image_paths[self.index]
            return os.path.basename(image_path)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if kwargs['mode'] != 'single_image_from_folder':
            return float("NaN")
        else:
            fl = FRED_LoadImage_V3.BatchImageLoader(kwargs['path'])
            filename = fl.get_current_image()
            image = os.path.join(kwargs['path'], filename)
            sha = get_sha256(image)
            return sha

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True

# Dictionary mapping node names to their corresponding classes
NODE_CLASS_MAPPINGS = {
    "FRED_LoadImage_V3": FRED_LoadImage_V3
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_LoadImage_V3": "👑 FRED_LoadImage_V3"
=======
import os
from PIL import Image, ImageOps, ImageSequence
import numpy as np
import torch
import hashlib
import folder_paths
import node_helpers
import cv2
import glob
import random
import fnmatch

ALLOWED_EXT = ('.jpeg', '.jpg', '.png', '.tiff', '.gif', '.bmp', '.webp')

# SHA-256 Hash
def get_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as file:
        for chunk in iter(lambda: file.read(4096), b''):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

class FRED_LoadImage_V3:
    def __init__(self):
        self.current_index = 0  # Initialize the index

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
                "mode": (["no_folder", "single_image_from_folder", "incremental_image_from_folder", "random_from_folder"],),
                "index": ("INT", {"default": 0, "min": 0, "max": 150000, "step": 1}),
                "path": ("STRING", {"default": '', "multiline": False}),
            },
            "optional": {
                "filename_text_extension": (["true", "false"], {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT", "INT", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = (
        "IMAGE", 
        "MASK", 
        "IMAGE_SIZE_KB", 
        "WIDTH", 
        "HEIGHT", 
        "QUALITY_SCORE", 
        "IMAGES QUANTITY IN FOLDER", 
        "SNR", 
        "FOLDER_PATH", 
        "filename_text"
    )
    FUNCTION = "load_image"
    CATEGORY = "FRED/image"
    
    def load_image(self, image, mode="no_folder", index=0, path="", filename_text_extension="false"):
        # Initialize image_path
        image_path = None
        # print("folder path:", path)

        if mode == "no_folder":
            # Execute normally for mode "no_folder"
            if isinstance(image, str):
                image_path = folder_paths.get_annotated_filepath(image)
                img = node_helpers.pillow(Image.open, image_path)
                filename = os.path.basename(image_path)
            elif isinstance(image, Image.Image):
                img = image
                filename = "direct_image_input"
            else:
                raise ValueError("Invalid image input type.")
        else:
            # Check if path is empty and handle accordingly
            if not path:
                print("No folder path provided, returning default image.")
                return self.return_default_image()

            if not os.path.exists(path):
                print(f"The path '{path}' does not exist. Returning default image.")
                return self.return_default_image()

            fl = self.BatchImageLoader(path)

            # Handle modes accordingly
            if mode == 'single_image_from_folder':
                img, filename = fl.get_image_by_id(index)
                # index += 1

            elif mode == 'incremental_image_from_folder':
                img, filename = fl.get_image_by_id(self.current_index)
                self.current_index += 1
                if self.current_index >= len(fl.image_paths):
                    self.current_index = 0

            else:  # random_from_folder
                # newindex = random.randint(0, len(fl.image_paths) - 1)
                # img, filename = fl.get_image_by_id(newindex)
                self.current_index = random.randint(0, len(fl.image_paths) - 1)
                img, filename = fl.get_image_by_id(self.current_index)

            if img is None:
                print("No valid image found, returning default image.")
                return self.return_default_image()

            image_path = os.path.join(path, filename)

        # Process the image
        output_images = []
        output_masks = []
        w, h = 0, 0 # Initialize with default values

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]
            
            if image.size[0] != w or image.size[1] != h:
                continue
            
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        # Retrieve image size
        # image_size_kb = int(self.calculate_image_size_in_kb(image_path)) if isinstance(image, str) else 0
        # print("image path:", image_path)
        image_size_kb = int(self.calculate_image_size_in_kb(image_path))
        _, height, width, _ = output_image.shape
        # print("is image instance:", isinstance(image, str))

        # Retrieve file name
        if filename_text_extension == "true":
            filename_text = filename
        else:
            filename_text = os.path.splitext(filename)[0]

        image_format = os.path.splitext(filename)[1]

        # Calculate image quality score
        quality_score = self.calculate_image_quality_score(image_size_kb, width, height, image_format)

        noise_level, snr_value = self.calculate_image_noise(image)

        # Check if the provided path is a directory only for folder modes
        if mode != "no_folder" and not os.path.isdir(path):
            raise ValueError(f"The path '{path}' is not a valid directory.")
        
        # List of valid image extensions
        valid_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.webp']

        images_count = 0

        # Iterate through each file in the directory only for folder modes
        if mode != "no_folder":
            # for filename in os.listdir(image_path):
            for filename in os.listdir(path):
                if any(fnmatch.fnmatch(filename, ext) for ext in valid_extensions):
                    # full_path = os.path.join(image_path, filename)
                    full_path = os.path.join(path, filename)
                    if os.path.isfile(full_path):
                        images_count += 1

        return (output_image, output_mask, image_size_kb, width, height, quality_score, images_count, snr_value, path, filename_text)

    def return_default_image(self):
        # Logic to create or load a default image
        default_image = np.zeros((64, 64, 3), dtype=np.float32)  # Example: a blank image
        default_image_tensor = torch.from_numpy(default_image)[None, :]  # Add batch dimension
        default_mask = torch.zeros((1, 64, 64), dtype=torch.float32)  # Example: a blank mask
        image_size_kb = 0  # Default size
        width, height = 64, 64  # Default dimensions
        quality_score = 0  # Default quality score
        images_count = 0  # Default image count
        snr_value = 0  # Default SNR value
        filename_text = "default_image"

        return (default_image_tensor, default_mask, image_size_kb, width, height, quality_score, images_count, snr_value, filename_text)

    def calculate_image_quality_score(self, image_size_kb, width, height, image_format):
        # Taille non compressée en Ko
        uncompressed_size = (width * height * 3) / 1024  # Converti à Ko
        uncompressed_size = int(uncompressed_size)  # Assurez-vous que c'est un entier

        # Calculer la taille de l'image en pixels
        pixel_size = width * height

        # Éviter la division par zéro
        if uncompressed_size == 0 or pixel_size == 0:
            return 0

        # Calculer le score basé sur la taille de l'image et la taille du fichier
        if image_size_kb >= uncompressed_size:
            score = 100
        else:
            score = int((image_size_kb / uncompressed_size) * 100)

        # Ajuster le score en fonction du format d'image
        if image_format.lower() in ('jpeg', 'jpg'):
            score *= 0.8 
        elif image_format.lower() == 'png':
            score *= 1.1 
        elif image_format.lower() == 'webp':
            score *= 1.2  # Prime pour la compression efficace de WebP

        # Ajuster le score en fonction de la taille de l'image
        score_adjusted = score * (pixel_size / (width * height))

        return max(0, min(100, score_adjusted))  # Assurez-vous que le score reste entre 0 et 100

    def calculate_image_size_in_kb(self, image_path):
        file_size_in_bytes = os.path.getsize(image_path)
        file_size_in_kb = file_size_in_bytes / 1024  # Convert bytes to kilobytes
        return file_size_in_kb

    def calculate_image_noise(self, image_tensor):
        """
        Calcule le niveau de bruit dans une image à l'aide de la variance locale et du SNR.
        
        Args:
            image_tensor: torch.Tensor, image en format PyTorch.

        Returns:
            noise_level: float, estimation du niveau de bruit par la variance locale.
            snr: float, rapport signal/bruit.
        """
        # Convertir le tenseur PyTorch en tableau NumPy
        image_np = image_tensor.squeeze().cpu().numpy() * 255.0
        image_np = image_np.astype(np.uint8)

        # Vérifier si l'image est dans le bon format (H, W, C)
        if image_np.shape[0] in [1, 3]:
            # Si c'est le cas, permuter pour obtenir (H, W, C)
            image_np = np.transpose(image_np, (1, 2, 0))

        # Si l'image a 3 canaux, on la convertit en niveau de gris
        if image_np.shape[2] == 3:
            image_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("L'image n'a pas un format de canaux valide : {}".format(image_np.shape))

        # Variance locale comme mesure de bruit
        mean = cv2.blur(image_gray, (3, 3))  # Moyenne locale
        diff = image_gray - mean
        variance = np.mean(diff ** 2)  # Variance locale des pixels voisins (bruit)

        # Calcul du SNR (Signal-to-Noise Ratio)
        signal_power = np.mean(image_gray ** 2)  # Puissance du signal (moyenne des carrés des pixels)
        noise_power = variance  # Puissance du bruit estimée par la variance locale
        snr = 10 * np.log10(signal_power / noise_power) if noise_power != 0 else float('inf')
        
        return variance, snr

    class BatchImageLoader:
        def __init__(self, directory_path):
            self.image_paths = []
            self.index = 0
            self.load_images(directory_path)
            self.image_paths.sort()

        def load_images(self, directory_path):
            for root, _, files in os.walk(directory_path):
                for file_name in files:
                    if file_name.lower().endswith(ALLOWED_EXT):
                        abs_file_path = os.path.abspath(os.path.join(root, file_name))
                        self.image_paths.append(abs_file_path)

        def get_image_by_id(self, image_id):
            while image_id < len(self.image_paths):
                try:
                    i = Image.open(self.image_paths[image_id])
                    i = ImageOps.exif_transpose(i)
                    return i, os.path.basename(self.image_paths[image_id])
                except (OSError, IOError):
                    image_id += 1
                    print(f"Skipping invalid image at index `{image_id}`")
            return None, None

        def get_next_image(self):
            while self.index < len(self.image_paths):
                try:
                    image_path = self.image_paths[self.index]
                    self.index += 1
                    if self.index == len(self.image_paths):
                        self.index = 0
                    i = Image.open(image_path)
                    i = ImageOps.exif_transpose(i)
                    return i, os.path.basename(image_path)
                except (OSError, IOError):
                    print(f"Skipping invalid image at index `{self.index}`")
                    self.index += 1
            return None, None

        def get_current_image(self):
            if self.index >= len(self.image_paths):
                self.index = 0
            image_path = self.image_paths[self.index]
            return os.path.basename(image_path)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # if kwargs['mode'] != 'single_image_from_folder':
        if kwargs['mode'] in ('incremental_image_from_folder', 'random_from_folder'):
            return float("NaN")
        else:
            fl = FRED_LoadImage_V3.BatchImageLoader(kwargs['path'])
            filename = fl.get_current_image()
            image = os.path.join(kwargs['path'], filename)
            sha = get_sha256(image)
            return sha

    # @classmethod
    # def VALIDATE_INPUTS(s, image):
        # if not folder_paths.exists_annotated_filepath(image):
            # return "Invalid image file: {}".format(image)
        # return True

# Dictionary mapping node names to their corresponding classes
NODE_CLASS_MAPPINGS = {
    "FRED_LoadImage_V3": FRED_LoadImage_V3
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_LoadImage_V3": "👑 FRED_LoadImage_V3"
>>>>>>> master
}