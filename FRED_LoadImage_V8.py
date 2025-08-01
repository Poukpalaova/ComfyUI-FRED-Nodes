# -*- coding: utf-8 -*-
# Updated version of FRED_LoadImage_V8.py with improved index handling, error management, and max image counting.
from __future__ import annotations

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
import pillow_avif  # must be imported before PIL.Image
from PIL import Image, ImageOps, ImageSequence

import sys
from comfy.comfy_types.node_typing import ComfyNodeABC, InputTypeDict, IO

ALLOWED_EXT = ('.jpeg', '.jpg', '.png', '.tiff', '.gif', '.bmp', '.webp', '.avif')
CACHE_VERSION = 2

HELP_MESSAGE = f"""
🖼️ FRED_LoadImage_V8 — Image Loader with Index Management, Preview & Logging

📂 LOAD MODES:
- 📁 *image_from_folder*: Loads an image from a folder using the given index.
- 📄 *no_folder*: Accepts direct image input or image filename from the ComfyUI input list.

🖼️ IMAGE PREVIEW:
- `show_image_from_path`: If enabled, the loaded image is saved in ComfyUI’s temp folder and is displayed in this node:
  → `{folder_paths.get_temp_directory()}`
  These images are automatically **deleted when ComfyUI restarts**.
  ⚠️ Avoid enabling this option during large batch runs as it may slow down execution.

🔢 INDEX CONTROL:
- `index`: The image index to load from the folder (0-based). To get an index mode, Use primitive to include:
  - `fixed`: Uses the same index repeatedly.
  - `increment`: Advances the index at each run.
  - `decrement`: Decreases the index.
  - `randomize`: Chooses a random index.
- `stop_at_max_index`: Prevents loading beyond this maximum index (useful for batching or iteration).
- `next_index_if_any_invalid`: 
  - `if false`: node will raise an exception if an invalid path or image is loaded.
  - `if true`: will skip to next valid path or image, but all the index in between will output the current valid image.

📁 PATH:
- `root path of where you want to load images.

📁 SUBFOLDERS:
- `include_subdirectories`: When enabled, all subfolders are recursively scanned for images.

🧹 CLEAR CACHE & LOGS:
- `clear_cache`: When enabled:
  - Deletes the image index cache (used to speed up folder scanning)
  - Deletes the log of invalid images (see below)
  This operation occurs **immediately**, and cannot be undone.

🚫 INVALID IMAGES LOGGING:
- Invalid or unreadable images are skipped and their full paths are logged.
- These paths are stored in a persistent file:
  → `{os.path.join(os.path.expanduser("~"), ".fred_logs", "invalid_images_log.txt")}`
- The log **accumulates over time** (unless cleared via `clear_cache`).
- The content is also returned in the `skipped_report` output for inspection.

📄 OUTPUTS:
- `image`, `mask`, `width`, `height`
- `index`: current valid image index
- `total_images_in_folder`: can include all subfolders images when include_subdirectories is true
- `image_count_in_current_folder`
- `folder_path`, `current_folder_path`, `filename_text`
- `skipped_report`: A newline-separated list of invalid image paths (accumulated)
- `help`: This message

🔧 DEBUG TIPS:
- Keep an eye on `skipped_report` to track folders or files to clean up.
- Use `clear_cache=True` once in a while to reset both image cache and error logs.

"""

class _FRED_PreviewHelper:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_image_preview_" + ''.join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))

    def get_unique_filename(self, filename_prefix):
        os.makedirs(self.output_dir, exist_ok=True)
        counter = 1
        while True:
            file = f"{filename_prefix}{self.prefix_append}_{counter:04d}.png"
            full_path = os.path.join(self.output_dir, file)
            if not os.path.exists(full_path):
                return full_path, file
            counter += 1

    def save_image(self, image_tensor, filename_prefix):
        results = []
        try:
            full_path, file = self.get_unique_filename(filename_prefix)
            img = image_tensor[0].clamp(0, 1).mul(255).byte().cpu().numpy()
            img = Image.fromarray(img)
            img.save(full_path)
            results.append({"filename": file, "subfolder": "", "type": self.type})
        except Exception as e:
            print(f"[Preview Error] {e}")
        return {"images": results}

# class FRED_LoadImage_V8:
class FRED_LoadImage_V8(ComfyNodeABC):
    @classmethod
    # def INPUT_TYPES(s):
    def INPUT_TYPES(cls) -> InputTypeDict:
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(ALLOWED_EXT)]
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
                "mode": (["no_folder", "image_from_folder"],),
                "show_image_from_path": ("BOOLEAN", {"default": False}),
                # "index": ("INT", {"default": 0, "min": 0, "max": 999999999, "step": 1}),
                "index": (IO.INT, {"default": 0, "min": 0, "max": 999999999, "control_after_generate": True}),
                "stop_at_max_index": ("INT", {"default": 999999999, "min": 0, "max": 999999999, "step": 1}),
                "next_index_if_any_invalid": ("BOOLEAN", {"default": True}),
                "path": ("STRING", {"default": '', "multiline": False}),
                "include_subdirectories": ("BOOLEAN", {"default": False}),
                "clear_cache": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "filename_text_extension": (["true", "false"], {"default": "false"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", IO.INT, "INT", "INT", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "WIDTH", "HEIGHT", "INDEX", "TOTAL IMAGES QTY IN FOLDER(S)", "IMAGES QTY IN CURRENT FOLDER", "FOLDER_PATH", "FULL_FOLDER_PATH", "filename_text", "skipped_report", "help")

    FUNCTION = "load_image"
    CATEGORY = "👑FRED/image"

    def load_image(self, image, mode="no_folder", index=0, stop_at_max_index=999999999,
                   next_index_if_any_invalid=True, path="", include_subdirectories=False,
                   clear_cache=False, show_image_from_path=False, filename_text_extension="false", _skipped_list=None):

        if clear_cache:
            self._clear_invalid_log()
            _skipped_list = []  # ⚠️ reset mémoire aussi
        else:
            if _skipped_list is None:
                _skipped_list = self._load_skipped_list()

        if index >= stop_at_max_index:
            raise RuntimeError(f"Reached stop_at_max_index limit: index={index}, stop_at_max_index={stop_at_max_index}")

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
            width, height = img.size
            filename_text = filename if filename_text_extension == "true" else os.path.splitext(filename)[0]

            return (output_image, output_mask, width, height, index, 1, 1, os.path.dirname(image_path) if image_path else "", full_folder_path, filename_text, HELP_MESSAGE)

        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Invalid path provided: {path}")

        # if clear_cache:
            # self._clear_invalid_log()
            # # self._clear_image_cache(path, include_subdirectories)

        fl = self.BatchImageLoader(path, include_subdirectories, clear_cache=clear_cache)
        max_value = fl.get_total_image_count()

        if max_value == 0:
            raise RuntimeError("No valid images found in folder.")

        skipped_report = "None"

        try:
            image_path = fl.get_image_path_by_id(index)
            img = Image.open(image_path)
            img = ImageOps.exif_transpose(img)
        except Exception as e:
            # Sauvegarde l'image échouée ABSOLUE dès maintenant
            failed_path = os.path.abspath(image_path)

            if failed_path not in _skipped_list:
                _skipped_list.append(failed_path)
                self._log_invalid_path(failed_path)

            if next_index_if_any_invalid:
                index += 1
                result = self.load_image(
                    image, mode, index, stop_at_max_index,
                    next_index_if_any_invalid, path, include_subdirectories,
                    clear_cache, show_image_from_path, filename_text_extension,
                    _skipped_list=_skipped_list
                )

                if isinstance(result, dict):
                    image_data = result["result"]
                    ui = result.get("ui", {})
                else:
                    image_data = result
                    ui = {}

                final_result = (
                    image_data[0], image_data[1], image_data[2], image_data[3], image_data[4],
                    image_data[5], image_data[6],
                    path, image_data[8], image_data[9],
                    "\n".join(_skipped_list),
                    image_data[11]
                )
                return {
                    **ui,
                    "result": final_result
                }

            else:
                raise RuntimeError(f"Error loading image: {failed_path} → {e}")

        filename = os.path.basename(image_path)
        current_folder_path = os.path.dirname(image_path)

        image = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        mask = torch.ones((1, img.size[1], img.size[0]), dtype=torch.float32)

        width, height = img.size
        filename_text = filename if filename_text_extension == "true" else os.path.splitext(filename)[0]

        skipped_report = "\n".join(sorted(set(_skipped_list))) if _skipped_list else "None"

        ui_result = {}
        if show_image_from_path:
            previewer = _FRED_PreviewHelper()
            ui_result = {"ui": previewer.save_image(image, "loaded_image_preview")}

        return {
            **ui_result,
            "result": (
                image, mask, width, height, index, max_value,
                self.count_images_in_current_folder(current_folder_path),
                path, current_folder_path, filename_text,
                skipped_report,
                HELP_MESSAGE + ("[Cache] Rebuilt." if clear_cache else "")
            )
        }

    def count_images_in_current_folder(self, path):
        valid = 0
        for f in os.listdir(path):
            if f.lower().endswith(ALLOWED_EXT):
                full = os.path.join(path, f)
                if os.path.isfile(full):
                    try:
                        Image.open(full).verify()
                        valid += 1
                    except:
                        continue
        return valid

    def _log_invalid_path(self, path):
        try:
            log_dir = os.path.join(os.path.expanduser("~"), ".fred_logs")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "invalid_images_log.txt")

            # Éviter les doublons disque
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8") as f:
                    if path in f.read():
                        return  # déjà logué

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(path + "\n")
        except Exception as e:
            print(f"[Invalid Image Log Error] {e}")

    def _load_skipped_list(self):
        try:
            log_dir = os.path.join(os.path.expanduser("~"), ".fred_logs")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "invalid_images_log.txt")
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8") as f:
                    return list(set(line.strip() for line in f if line.strip()))
        except Exception as e:
            print(f"[Skipped List Load Error] {e}")
        return []

    def _clear_invalid_log(self):
        try:
            log_path = os.path.join(os.path.expanduser("~"), ".fred_logs", "invalid_images_log.txt")
            if os.path.exists(log_path):
                os.remove(log_path)
                print("[FRED] Cleared invalid image log.")
        except Exception as e:
            print(f"[FRED] Failed to clear invalid image log: {e}")

    class BatchImageLoader:
        # def __init__(self, directory_path, include_subdirectories=False, help_log=None, clear_cache=False):
        def __init__(self, directory_path, include_subdirectories=False, help_log=None, clear_cache=False):
            self.directory_path = directory_path
            self.include_subdirectories = include_subdirectories
            self.image_paths = []
            self.cache_file = self._get_cache_path()
            self.dir_hash = self._calculate_directory_hash()
            if clear_cache or not self._load_from_cache():
                # if help_log is not None:
                    # help_log.append("[Cache] Rebuilding image index cache for this folder.")
                self._build_index()
                self._save_to_cache()

        def _get_cache_path(self):
            temp_dir = os.path.join(os.path.expanduser("~"), ".fred_image_cache")
            os.makedirs(temp_dir, exist_ok=True)
            folder_name = os.path.basename(os.path.normpath(self.directory_path))
            full_identifier = f"{self.directory_path}|{folder_name}|{str(self.include_subdirectories)}"
            safe_name = hashlib.md5(full_identifier.encode()).hexdigest()
            return os.path.join(temp_dir, f"{safe_name}.json")

        def _calculate_directory_hash(self):
            try:
                folder_mtime = str(os.path.getmtime(self.directory_path))
                base = self.directory_path + str(self.include_subdirectories) + folder_mtime
            except:
                base = self.directory_path + str(self.include_subdirectories)
            return hashlib.md5(base.encode()).hexdigest()

        def _load_from_cache(self):
            if os.path.exists(self.cache_file):
                try:
                    with open(self.cache_file, 'r') as f:
                        data = json.load(f)
                        if data.get('dir_hash') == self.dir_hash:
                            self.image_paths = data['image_paths']
                            return True
                except:
                    return False
            return False

        def _save_to_cache(self):
            with open(self.cache_file, 'w') as f:
                json.dump({'dir_hash': self.dir_hash, 'image_paths': self.image_paths, 'timestamp': time.time()}, f)

        def _build_index(self):
            self.image_paths = []
            for root, _, files in os.walk(self.directory_path) if self.include_subdirectories else [(self.directory_path, [], os.listdir(self.directory_path))]:
                for file in files:
                    if file.lower().endswith(ALLOWED_EXT):
                        full = os.path.abspath(os.path.join(root, file))
                        self.image_paths.append(full)
            self.image_paths.sort()

        def get_total_image_count(self):
            return len(self.image_paths)

        def get_image_path_by_id(self, image_id):
            if not self.image_paths:
                return None
            return self.image_paths[image_id % len(self.image_paths)]

    # @classmethod
    # def IS_CHANGED(cls, **kwargs):
        # if kwargs.get("clear_cache", False):
            # return True
        # key = f"{kwargs.get('image', '')}_{kwargs.get('mode', '')}_{kwargs.get('index', '')}_{kwargs.get('path', '')}_{kwargs.get('include_subdirectories', False)}"
        # return hashlib.sha256(key.encode()).hexdigest()
    @staticmethod
    def IS_CHANGED(*args, **kwargs):
        return False

    @classmethod
    def VALIDATE_INPUTS(cls, image, mode="no_folder", path="", **kwargs):
        if mode == "no_folder":
            if not folder_paths.exists_annotated_filepath(image):
                return f"Image not found: {image}"
        elif mode == "image_from_folder":
            if not path or not os.path.exists(path):
                return f"Invalid folder path: {path}"
        return True


NODE_CLASS_MAPPINGS = {
    "FRED_LoadImage_V8": FRED_LoadImage_V8
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_LoadImage_V8": "👑 FRED Load Image V8 (avif support)"
}