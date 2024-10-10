<<<<<<< HEAD
import os
from PIL import Image, ImageOps, ImageSequence
import numpy as np
import torch
import hashlib
import folder_paths
import node_helpers
import cv2

class FRED_LoadImage_V2:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
                "required": {
                    "image": (sorted(files), {"image_upload": True}),
                },
                "optional": {
                    "filename_text_extension": (["true", "false"], {"default": False}),
                }
            }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "IMAGE", 
        "MASK", 
        "IMAGE_SIZE_KB", 
        "WIDTH", 
        "HEIGHT", 
        "QUALITY_SCORE", 
        "SNR", 
        "filename_text"
    )
    FUNCTION = "load_image"
    CATEGORY = "FRED/image"
    def load_image(self, image, filename_text_extension="true"):
        image_path = folder_paths.get_annotated_filepath(image)
        
        img = node_helpers.pillow(Image.open, image_path)
        
        output_images = []
        output_masks = []
        w, h = None, None

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
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        # retreive image size
        image_size_kb = int(self.calculate_image_size_in_kb(image_path))
        _, height, width, _ = output_image.shape

        # retreive file name
        if filename_text_extension == "true":
            filename = os.path.basename(image_path)
        else:
            filename = os.path.splitext(os.path.basename(image_path))[0]

        # calculate image quality score
        uncompressed_size = int((width * height * 3) / 1024)  # Convert to KB and ensure int
        if image_size_kb >= uncompressed_size:
            score = 100
        else:
            score = int((image_size_kb / uncompressed_size) * 100)

        noise_level, snr_value = self.calculate_image_noise(image)

        return (output_image, output_mask, width, height, image_size_kb, score, snr_value, filename)

    def calculate_image_size_in_kb(self, image_path):
        file_size_in_bytes = os.path.getsize(image_path)
        file_size_in_kb = file_size_in_bytes / 1024  # Convert bytes to kilobytes
        return file_size_in_kb
        
#    @classmethod
#    def IS_CHANGED(s, image):
#        image_path = folder_paths.get_annotated_filepath(image)
#        m = hashlib.sha256()
#        with open(image_path, 'rb') as f:
#            m.update(f.read())
#        return m.digest().hex()

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

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True

# Dictionary mapping node names to their corresponding classes
NODE_CLASS_MAPPINGS = {
    "FRED_LoadImage_V2": FRED_LoadImage_V2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_LoadImage_V2": "👑 FRED_LoadImage_V2"
=======
import os
from PIL import Image, ImageOps, ImageSequence
import numpy as np
import torch
import hashlib
import folder_paths
import node_helpers
import cv2

class FRED_LoadImage_V2:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
                "required": {
                    "image": (sorted(files), {"image_upload": True}),
                },
                "optional": {
                    "filename_text_extension": (["true", "false"], {"default": False}),
                }
            }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "IMAGE", 
        "MASK", 
        "IMAGE_SIZE_KB", 
        "WIDTH", 
        "HEIGHT", 
        "QUALITY_SCORE", 
        "SNR", 
        "filename_text"
    )
    FUNCTION = "load_image"
    CATEGORY = "FRED/image"
    def load_image(self, image, filename_text_extension="true"):
        image_path = folder_paths.get_annotated_filepath(image)
        
        img = node_helpers.pillow(Image.open, image_path)
        
        output_images = []
        output_masks = []
        w, h = None, None

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
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        # retreive image size
        image_size_kb = int(self.calculate_image_size_in_kb(image_path))
        _, height, width, _ = output_image.shape

        # retreive file name
        if filename_text_extension == "true":
            filename = os.path.basename(image_path)
        else:
            filename = os.path.splitext(os.path.basename(image_path))[0]

        # calculate image quality score
        uncompressed_size = int((width * height * 3) / 1024)  # Convert to KB and ensure int
        if image_size_kb >= uncompressed_size:
            score = 100
        else:
            score = int((image_size_kb / uncompressed_size) * 100)

        noise_level, snr_value = self.calculate_image_noise(image)

        return (output_image, output_mask, width, height, image_size_kb, score, snr_value, filename)

    def calculate_image_size_in_kb(self, image_path):
        file_size_in_bytes = os.path.getsize(image_path)
        file_size_in_kb = file_size_in_bytes / 1024  # Convert bytes to kilobytes
        return file_size_in_kb
        
#    @classmethod
#    def IS_CHANGED(s, image):
#        image_path = folder_paths.get_annotated_filepath(image)
#        m = hashlib.sha256()
#        with open(image_path, 'rb') as f:
#            m.update(f.read())
#        return m.digest().hex()

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

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True

# Dictionary mapping node names to their corresponding classes
NODE_CLASS_MAPPINGS = {
    "FRED_LoadImage_V2": FRED_LoadImage_V2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_LoadImage_V2": "👑 FRED_LoadImage_V2"
>>>>>>> master
}