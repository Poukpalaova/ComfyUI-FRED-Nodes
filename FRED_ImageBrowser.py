import torch
import os
from PIL import Image, ImageOps, ImageSequence
from PIL.PngImagePlugin import PngInfo
import numpy as np
import safetensors.torch
import folder_paths
from folder_paths import get_directory_by_type
from nodes import LoraLoader, CheckpointLoaderSimple
import importlib

import latent_preview

# class FRED_FolderSelector:
    # @classmethod
    # def INPUT_TYPES(cls):
        # """
        # Cette méthode de classe définit les entrées du nœud. Pour ce nœud, aucune entrée n'est nécessaire
        # puisqu'il sert à sélectionner un dossier parmi une liste prédéfinie.
        # """
        # return {}

    # @classmethod
    # def OUTPUT_TYPES(cls):
        # """
        # Cette méthode de classe définit les types de sortie du nœud.
        # """
        # return {"folder_name": "STRING"}

    # CATEGORY = "Utilities/Folder Selector"

    # RETURN_TYPES = ("STRING",)
    # RETURN_NAMES = ("folder_name",)
    # FUNCTION = "select_folder"

    # def select_folder(self):
        # """
        # Cette méthode serait utilisée pour afficher une liste de dossiers et permettre à l'utilisateur de sélectionner un.
        # Dans cet exemple, nous simulons simplement une sélection avec un nom de dossier statique.
        # """
        # p = os.path.dirname(os.path.realpath(__file__))
        # folder_path = os.path.join(p, 'img_lists')
        # folders = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
        
        # # Ici, vous pourriez permettre à l'utilisateur de choisir parmi les dossiers `folders`
        # # Pour l'exemple, nous retournons simplement le premier dossier s'il existe
        # selected_folder = folders[0] if folders else None
        
        # return selected_folder

# class FRED_ImageBrowser:
    # @classmethod
    # def INPUT_TYPES(cls):
        # # Predetermined folder is directly used here
        # default_folder = "dress"  # This is now directly specified
        # images = cls.get_images(default_folder)

        # return {
            # "required": {
                # "image": (images, {"default": images[0] if images else None}),  # Selection of an image
                # "weight": ("FLOAT", {
                    # "default": 1.2,
                    # "step": 0.05,
                    # "min": 0,
                    # "max": 1.75,
                    # "display": "slider",
                # }),
            # }
        # }

    # CATEGORY = "image/Load Image/Image with Prompt"
    # RETURN_TYPES = ("STRING", "IMAGE",)
    # RETURN_NAMES = ("name", "image",)
    # FUNCTION = "load_image"

    # @staticmethod
    # def get_images(folder):
        # # Function to list images in the predetermined folder
        # p = os.path.dirname(os.path.realpath(__file__))
        # image_dir = os.path.join(p, 'img_lists', folder)
        # return [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f)) and (f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg'))]

    # def load_image(self, image, weight=1):
        # # Load image from the predetermined folder
        # p = os.path.dirname(os.path.realpath(__file__))
        # image_path = os.path.join(p, 'img_lists', 'dress', image)
        
        # img = Image.open(image_path)
        # img = ImageOps.exif_transpose(img)
        # img = img.convert("RGB")
        # img_array = np.array(img).astype(np.float32) / 255.0
        # output_image = torch.from_numpy(img_array)[None,]

        # image_name = os.path.splitext(image)[0]
        # prompt = f"({image_name}:{round(weight, 2)})"

        # return (prompt, output_image)
        
# class FRED_ImageBrowser:
    # predetermined_folder = "dress"  # Static folder name

    # @classmethod
    # # def get_images(cls):
        # # base_dir = os.path.dirname(os.path.realpath(__file__))
        # # folder_path = os.path.join(base_dir, 'img_lists', cls.predetermined_folder)
        # # image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # # # Convert image filenames to the expected dictionary format for populate_items
        # # images = [{"content": image_file} for image_file in image_files]
        
        # # # Now populate_items will receive the correct data structure
        # # populate_items(images, 'image')
        # # return images
    # def select_image(self, image_name):
        # # Logic to handle image selection and possibly display the image
        # folder_path = "path/to/your/images/folder"
        # image_path = os.path.join(folder_path, image_name)
        # # Load the image or handle as needed for your application
        # return image_path  # Or return image data, depending on your requirements
        
    # @classmethod
    # def INPUT_TYPES(cls):
        # # Get the directory of the current script/file
        # current_dir = os.path.dirname(os.path.realpath(__file__))

        # # Construct the relative path to the "img_lists/dress" folder
        # folder_path = os.path.join(current_dir, "img_lists", "dress")
        # image_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        # populate_items(image_names, folder_path)  # Assume "images" is the type for folder path manipulation
        
        # return {
            # "required": {
                # "image_name": (image_names, {"default": image_names[0] if image_names else None}),
                # "weight": ("FLOAT", {
                    # "default": 1.2,
                    # "step": 0.05,
                    # "min": 0,
                    # "max": 1.75,
                    # "display": "slider",
                # }),
            # }
        # }

    # FUNCTION = "load_image"

    # def load_image(self, image, weight=1):
        # # Load image from the predetermined folder
        # p = os.path.dirname(os.path.realpath(__file__))
        # image_path = os.path.join(p, 'img_lists', 'dress', image)
        
        # img = Image.open(image_path)
        # img = ImageOps.exif_transpose(img)
        # img = img.convert("RGB")
        # img_array = np.array(img).astype(np.float32) / 255.0
        # output_image = torch.from_numpy(img_array)[None,]

        # image_name = os.path.splitext(image)[0]
        # prompt = f"({image_name}:{round(weight, 2)})"

        # return (prompt, output_image)
        
    # @classmethod
    # def IS_CHANGED(s, image):
        # image_path = self.get_img_path(image_name, "folder")
        # m = hashlib.sha256()
        # with open(image_path, 'rb') as f:
            # m.update(f.read())
        # return m.digest().hex()
        
# class FRED_ImageBrowser_Generic:
    # @classmethod
    # def INPUT_TYPES(cls):
        # # Assuming folder_paths.get_filename_list("checkpoints") retrieves checkpoint names
        # # names = folder_paths.get_filename_list("checkpoints")
        # current_dir = os.path.dirname(os.path.realpath(__file__))
        # # Construct the relative path to the "img_lists/dress" folder
        # folder_path = os.path.join(current_dir, 'img_lists/dress/')
        # # names = folder_paths.get_filename_list(os.path.join(folder_paths.base_path, "custom_nodes\\ComfyUI-FRED-Prompt-Guidance\\img_lists\\dress"))
        # # Enrich checkpoint names with images for a more interactive UI experience
        # # populate_items(names, "checkpoints")
        # # names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        # names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        # populate_items(names, "generic")
        
        # return {"required": { "images_name": (names, ) }}

    # RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    # FUNCTION = "load_checkpoint"
    # CATEGORY = "loaders"

    # def load_checkpoint(self, ckpt_name, output_vae=True, output_clip=True):
        # # Assuming folder_paths.get_full_path constructs the full path to the checkpoint
        # ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name["content"])
        
        # # Load the checkpoint with optional VAE and CLIP components
        # out = comfy.sd.load_checkpoint_guess_config(
            # ckpt_path, 
            # output_vae=output_vae, 
            # output_clip=output_clip, 
            # embedding_directory=folder_paths.get_folder_paths("embeddings")
        # )
        # return out[:3]
        
# class FRED_ImageBrowser_Dress:
    # @classmethod
    # def INPUT_TYPES(s):
        # current_dir = os.path.dirname(os.path.realpath(__file__))
        # # Construct the relative path to the "img_lists/dress" folder
        # folder_path = os.path.join(current_dir, 'img_lists/dress/')
        # names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        # max_float_value = 1.75
        # # populate_items(names, 'dress')

        # return {
            # "required": {
                # "image": (sorted(names), {"image_upload_dress": True}),
                # "weight": ("FLOAT", {
                    # "default": 1.2,
                    # "step": 0.05,
                    # "min": 0,
                    # "max": max_float_value,
                    # "display": "slider",
               # }),
            # }
        # }

    # CATEGORY = "image"
    # RETURN_NAMES = ("image", "name")
    # RETURN_TYPES = ("IMAGE", "STRING")
    # FUNCTION = "load_dress_images"

    # def load_dress_images(self, image, weight=1):
        # image_path = folder_paths.get_annotated_filepath(image)
        # # Extract image name here, using the file path
        # image_name, _ = os.path.splitext(os.path.basename(image_path))
        # prompt = f"({image_name}:{round(weight, 2)})"
        
        # img = Image.open(image_path)
        # output_images = []
        # for i in ImageSequence.Iterator(img):
            # i = ImageOps.exif_transpose(i)
            # if i.mode == 'I':
                # i = i.point(lambda i: i * (1 / 255))
            # image = i.convert("RGB")
            # image = np.array(image).astype(np.float32) / 255.0
            # image = torch.from_numpy(image)[None,]
            # output_images.append(image)
        
        # if len(output_images) > 1:
            # output_image = torch.cat(output_images, dim=0)
        # else:
            # output_image = output_images[0]

        # return (output_image, prompt)

    # # @classmethod
    # # def IS_CHANGED(s, image):
        # # image_path = folder_paths.get_annotated_filepath(image)
        # # m = hashlib.sha256()
        # # with open(image_path, 'rb') as f:
            # # m.update(f.read())
        # # return m.digest().hex()
        
    # @classmethod
    # def IS_CHANGED(s, image):
        # image_path = get_img_path(image_name, "dress")
        # m = hashlib.sha256()
        # with open(image_path, 'rb') as f:
            # m.update(f.read())
        # return m.digest().hex()

    # @classmethod
    # def VALIDATE_INPUTS(s, image):
        # image_path = folder_paths.get_annotated_filepath(image)
        # file_exists = os.path.exists(image_path)
        # print(f"Checking path: {image_path}, exists: {file_exists}")
        # if not file_exists:
            # return "Invalid image file: {}".format(image)
        # return True
        
# def populate_items(names, type):
    # for idx, item_name in enumerate(names):
        # file_name = os.path.splitext(item_name)[0]
        # file_path = folder_paths.get_full_path(type, item_name)

        # if file_path is None:
            # print(f"(pysssss:better_combos) Unable to get path for {type} {item_name}")
            # continue

        # file_path_no_ext = os.path.splitext(file_path)[0]
        # has_image = False
        # for ext in ["png", "jpg", "jpeg", "preview.png"]:
            # if os.path.isfile(file_path_no_ext + "." + ext):
                # item_image = f"{file_name}.{ext}"
                # has_image = True
                # break

        # names[idx] = {
            # "content": item_name,
            # "image": f"{type}/{item_image}" if has_image else None,
        # }
    # names.sort(key=lambda i: i["content"].lower())
class FRED_ImageBrowser_Dress:
    @classmethod
    def INPUT_TYPES(s):
        p = os.path.dirname(os.path.realpath(__file__))
        folder_path = os.path.join(p, 'img_lists/dress/')
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        max_float_value = 1.75

        return {
            "required": {
                "image": (sorted(files), {"image_upload_dress": True}),
                "weight": ("FLOAT", {
                    "default": 1.2,
                    "step": 0.05,
                    "min": 0,
                    "max": max_float_value,
                    "display": "slider",
               }),
            }
        }


    CATEGORY = "image"

    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("name", "image", "folder path")
    FUNCTION = "load_dress_images"

    def load_dress_images(self, image, weight=1):
        image_full_name = image
        image_name = image_full_name.rsplit('.', 1)[0]

        image_path =  get_img_path(image_name, "dress")
        img = Image.open(image_path)
        output_images = []
        output_masks = []
        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        prompt = []

        if weight > 0:
            P_dress = f"({image_name}:{round(weight, 2)})"
            prompt.append(P_dress)

        return (P_dress, output_image, image_path)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = get_img_path(image_name, "dress")
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()
        
class FRED_ImageBrowser_Hair_Style:
    @classmethod
    def INPUT_TYPES(s):
        p = os.path.dirname(os.path.realpath(__file__))
        folder_path = os.path.join(p, 'img_lists/hair_style/')
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        max_float_value = 1.75

        return {
            "required": {
                "image": (sorted(files), {"image_upload_hair_style": True}),
                "weight": ("FLOAT", {
                    "default": 1.2,
                    "step": 0.05,
                    "min": 0,
                    "max": max_float_value,
                    "display": "slider",
               }),
            }
        }


    CATEGORY = "image"

    RETURN_TYPES = ("STRING", "IMAGE",)
    RETURN_NAMES = ("name", "image",)
    FUNCTION = "load_hair_style_images"

    def load_hair_style_images(self, image, weight=1):
        image_full_name = image
        image_name = image_full_name.rsplit('.', 1)[0]

        image_path =  get_img_path(image_name, "hair_style")
        img = Image.open(image_path)
        output_images = []
        output_masks = []
        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        prompt = []

        if weight > 0:
            P_hair_style = f"({image_name}:{round(weight, 2)})"
            prompt.append(P_hair_style)

        return (P_hair_style, output_image,)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = get_img_path(image_name, "hair_style")
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

class FRED_ImageBrowser_Eyes_Color:
    @classmethod
    def INPUT_TYPES(s):
        p = os.path.dirname(os.path.realpath(__file__))
        folder_path = os.path.join(p, 'img_lists/eyes_color/')
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        max_float_value = 1.75

        return {
            "required": {
                "image": (sorted(files), {"image_upload_eyes_color": True}),
                "weight": ("FLOAT", {
                    "default": 1.2,
                    "step": 0.05,
                    "min": 0,
                    "max": max_float_value,
                    "display": "slider",
               }),
            }
        }


    CATEGORY = "image"

    RETURN_TYPES = ("STRING", "IMAGE",)
    RETURN_NAMES = ("name", "image",)
    FUNCTION = "load_eyes_color_images"

    def load_eyes_color_images(self, image, weight=1):
        image_full_name = image
        image_name = image_full_name.rsplit('.', 1)[0]

        image_path =  get_img_path(image_name, "eyes_color")
        img = Image.open(image_path)
        output_images = []
        # output_masks = []
        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            # if 'A' in i.getbands():
                # mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                # mask = 1. - torch.from_numpy(mask)
            # else:
                # mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            # output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            # output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            # output_mask = output_masks[0]

        prompt = []

        if weight > 0:
            P_eyes_color = f"({image_name}:{round(weight, 2)})"
            prompt.append(P_eyes_color)

        return (P_eyes_color, output_image,)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = get_img_path(image_name, "eyes_color")
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

class FRED_ImageBrowser_Top:
    @classmethod
    def INPUT_TYPES(s):
        p = os.path.dirname(os.path.realpath(__file__))
        folder_path = os.path.join(p, 'img_lists/top/')
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        max_float_value = 1.75

        return {
            "required": {
                "image": (sorted(files), {"image_upload_top": True}),
                "weight": ("FLOAT", {
                    "default": 1.2,
                    "step": 0.05,
                    "min": 0,
                    "max": max_float_value,
                    "display": "slider",
               }),
            }
        }


    CATEGORY = "image"

    RETURN_TYPES = ("STRING", "IMAGE",)
    RETURN_NAMES = ("name", "image",)
    FUNCTION = "load_top_images"

    def load_top_images(self, image, weight=1):
        image_full_name = image
        image_name = image_full_name.rsplit('.', 1)[0]

        image_path =  get_img_path(image_name, "top")
        img = Image.open(image_path)
        output_images = []
        # output_masks = []
        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            # if 'A' in i.getbands():
                # mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                # mask = 1. - torch.from_numpy(mask)
            # else:
                # mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            # output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            # output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            # output_mask = output_masks[0]

        prompt = []

        if weight > 0:
            P_top = f"({image_name}:{round(weight, 2)})"
            prompt.append(P_top)

        return (P_top, output_image,)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = get_img_path(image_name, "top")
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

class FRED_ImageBrowser_Hair_Color:
    @classmethod
    def INPUT_TYPES(s):
        p = os.path.dirname(os.path.realpath(__file__))
        folder_path = os.path.join(p, 'img_lists/hair_color/')
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        max_float_value = 1.75

        return {
            "required": {
                "image": (sorted(files), {"image_upload_hair_color": True}),
                "weight": ("FLOAT", {
                    "default": 1.2,
                    "step": 0.05,
                    "min": 0,
                    "max": max_float_value,
                    "display": "slider",
               }),
            }
        }


    CATEGORY = "image"

    RETURN_TYPES = ("STRING", "IMAGE",)
    RETURN_NAMES = ("name", "image",)
    FUNCTION = "load_hair_color_images"

    def load_hair_color_images(self, image, weight=1):
        image_full_name = image
        image_name = image_full_name.rsplit('.', 1)[0]

        image_path =  get_img_path(image_name, "hair_color")
        img = Image.open(image_path)
        output_images = []
        output_masks = []
        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        prompt = []

        if weight > 0:
            P_hair_color = f"({image_name}:{round(weight, 2)})"
            prompt.append(P_hair_color)

        return (P_hair_color, output_image,)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = get_img_path(image_name, "hair_color")
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

def get_img_path(template_name, template_type):
    p = os.path.dirname(os.path.realpath(__file__))

    if os.name == 'posix':  # Unix/Linux/macOS
        separator = '/'
    elif os.name == 'nt':  # Windows
        separator = '\\'
    else:
        separator = '/'

    image_path = os.path.join(p, 'img_lists', template_type)
    image_filename = f"{template_name}.png"

    full_image_path = image_path + separator + image_filename

    return full_image_path
        
# Dictionary mapping node names to their corresponding classes
NODE_CLASS_MAPPINGS = {
    "FRED_ImageBrowser_Dress": FRED_ImageBrowser_Dress,
    "FRED_ImageBrowser_Hair_Style": FRED_ImageBrowser_Hair_Style,
    "FRED_ImageBrowser_Eyes_Color": FRED_ImageBrowser_Eyes_Color,
    "FRED_ImageBrowser_Top": FRED_ImageBrowser_Top,
    "FRED_ImageBrowser_Hair_Color": FRED_ImageBrowser_Hair_Color
    # "FRED_ImageBrowser_Generic": FRED_ImageBrowser_Generic,
    # "FRED_FolderSelector": FRED_FolderSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_ImageBrowser_Dress": "👑 FRED_ImageBrowser_Dress",
    "FRED_ImageBrowser_Hair_Style": "👑 FRED_ImageBrowser_Hair_Style",
    "FRED_ImageBrowser_Eyes_Color": "👑 FRED_ImageBrowser_Eyes_Color",
    "FRED_ImageBrowser_Top": "👑 FRED_ImageBrowser_Top",
    "FRED_ImageBrowser_Hair_Color": "👑 FRED_ImageBrowser_Hair_Color"
    # "FRED_ImageBrowser_Generic": "👑 FRED_ImageBrowser_Generic",
    # "FRED_FolderSelector": "👑 FRED_FolderSelector",
}