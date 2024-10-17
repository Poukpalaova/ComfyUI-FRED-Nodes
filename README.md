# FRED Nodes

This repository contains custom nodes for ComfyUI. This is a work in progress (WIP).

## Nodes

### FRED_AutoCropImage_SDXL_Ratio_v4
Description: Automatically crops and resizes images to fit SDXL aspect ratios. Features include auto-finding SDXL resolution, custom aspect ratios, various cropping options, resizing with different interpolation modes, prescaling, and preview generation.

### FRED_CropFace
Description: Detects and crops faces from images using the RetinaFace model. It provides options for confidence threshold, margin adjustment, and face selection. The node also generates a preview with detected faces highlighted.

### FRED_LoadImage_V5
Description: Loads and processes images for use in image generation pipelines. Supports single images or batches from a specified folder, handles various image formats, processes RGBA images, calculates image quality scores, and provides options for seed-based selection.

### FRED_LoadPathImagesPreview_v2
Description: Loads and previews multiple images from a specified path. It allows pattern matching for file selection and returns the count of images in the folder.

### FRED_LoadRetinaFace.py
Description: Loads the RetinaFace face detection model. It initializes the model using the "retinaface_resnet50" architecture and returns the loaded model for use in face detection tasks.
Description: [Your description here] Note: Same as [specify the similar node]

### FRED_photo_prompt.py (WIP)
Description: Generates photo prompts based on various style elements. It reads data from a JSON file and allows users to select different aspects of a photo prompt, including style, subject, framing, background, lighting, camera properties, and more.
Description: [Your description here] Note: Same as [specify the similar node]

### FRED_PreviewOnly.py (TEST)
Description: Extends the PreviewImage node to provide image preview functionality. It processes input images and sends them to the UI for preview, without modifying the original images.
Description: [Your description here]

### FRED_ImageBrowser (WIP)
Description: Image browsing and selection tool designed for integration with AI-powered image generation workflows. This module provides an intuitive interface for selecting and manipulating images across various categories, including dresses, hair styles, eye colors, tops, and hair colors.

### FRED_JoinImages
Description: Joins multiple images either vertically or horizontally. It can handle a single image or a list of images, and includes features for padding and resizing to ensure consistent dimensions.

### FRED_LoadImage_V2
Status: Deprecated

### FRED_LoadImage_V3
Status: Deprecated

### FRED_LoadImage_V4
Status: Deprecated
