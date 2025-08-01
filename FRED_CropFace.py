import os
import torch
import numpy as np
import cv2
from facexlib.detection import RetinaFace
from .utils import tensor2cv, cv2tensor, hex2bgr, BBox, models_dir

class FRED_CropFace:
    models_dir = os.path.join(models_dir, 'facexlib')
    def __init__(self):
        """Load RetinaFace model during node initialization"""
        from facexlib.detection import init_detection_model
        self.model = init_detection_model("retinaface_resnet50", model_rootpath=self.models_dir)
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # "model": ("RETINAFACE",),
                "image": ("IMAGE",),
                "confidence": ("FLOAT", {"default": 0.8, "min": 0, "max": 1}),
                "left_margin_factor": ("FLOAT", {"default": 0.6, "min": 0.0}),
                "right_margin_factor": ("FLOAT", {"default": 0.6, "min": 0.0}),
                "top_margin_factor": ("FLOAT", {"default": 0.4, "min": 0.0}),
                "bottom_margin_factor": ("FLOAT", {"default": 1, "min": 0.0}),
                "face_id": ("INT", {"default": 0, "min": 0}),
                "max_size": ("INT", {"default": 1536, "min": 256}), # Maximum size for detection
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "BBOX", "FLOAT", "FLOAT")
    RETURN_NAMES = ("face_image", "preview", "bbox", "face_pixel_ratio", "face_w_margin_pixel_ratio")
    FUNCTION = "crop"
    CATEGORY = "👑FRED/image/postprocessing"

    def crop(self, image: torch.Tensor, confidence: float, left_margin_factor: float, 
            right_margin_factor: float, top_margin_factor: float, bottom_margin_factor: float, 
            face_id: int, max_size: int):
        """Modified crop method using internal model"""
        img_cv = tensor2cv(image)
        img_height, img_width = img_cv.shape[:2]

        # Resize image for face detection if it's too large
        scale = 1
        if max(img_width, img_height) > max_size:
            scale = max_size / max(img_width, img_height)
            img_resized = cv2.resize(img_cv, (int(img_width * scale), int(img_height * scale)))
        else:
            img_resized = img_cv

        # Use self.model instead of passed parameter
        with torch.no_grad():
            bboxes = self.model.detect_faces(img_resized, confidence)

        if len(bboxes) == 0:
            print("WARNING! No face detected. Please adjust confidence or change picture. Input picture will be sent to output.")
            return image, image, np.zeros((4,)), 0.0

        # Scale bounding boxes back to original size
        bboxes = [(x0/scale, y0/scale, x1/scale, y1/scale, score, *[p/scale for p in points])
                  for (x0, y0, x1, y1, score, *points) in bboxes]
        bboxes = sorted(bboxes, key=lambda b: b[0])

        if face_id >= len(bboxes):
            print(f"ERROR! Invalid face_id: {face_id}. Total detected faces: {len(bboxes)}. Using face_id = 0.")
            face_id = 0

        # Create preview on original image
        detection_preview = self.visualize_detection(img_cv, bboxes)

        # Margin-augmented bbox (after add_margin)
        bboxes_with_margin = [
            self.add_margin(
                (int(min(x0, x1)), int(min(y0, y1)), int(abs(x1 - x0)), int(abs(y1 - y0))),
                left_margin_factor,
                right_margin_factor,
                top_margin_factor,
                bottom_margin_factor,
                img_width=img_width,
                img_height=img_height
            ) for (x0, y0, x1, y1, *_) in bboxes
        ]
        detection_preview = self.visualize_margin(detection_preview, bboxes_with_margin)

        x, y, w, h = bboxes_with_margin[face_id]
        face_margin_pixels = w * h  # Area with margin

        # Original face bbox (without margin)
        orig_bbox = bboxes[face_id]  # (x0, y0, x1, y1, ...)
        x0, y0, x1, y1 = map(int, orig_bbox[:4])
        face_w = abs(x1 - x0)
        face_h = abs(y1 - y0)
        face_pixels = face_w * face_h  # Area without margin

        total_pixels = img_height * img_width

        face_pixel_ratio = (face_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0
        face_w_margin_pixel_ratio = (face_margin_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0

        # Ensure crop stays within image bounds
        x1 = min(x + w, img_width)
        y1 = min(y + h, img_height)

        # Crop the face from the original image tensor
        # Assuming image shape is [B, H, W, C]
        cropped_face = image[0, y:y1, x:x1, :].unsqueeze(0)

        # bbox_list = []
        # bbox_list.append((x, y, w, h))

        return cropped_face, cv2tensor(detection_preview), bboxes_with_margin, face_pixel_ratio, face_w_margin_pixel_ratio

    # def crop_faces(self, bboxes, image: torch.Tensor):
        # """
        # Returns: list of Tensor[h, w, c] of faces
        # """
        # return [image[0, y:y+h, x:x+w, :] for (x, y, w, h) in bboxes]

    def visualize_margin(self, img, bboxes):
        img = np.copy(img)
        for bbox in bboxes:
            # x0, y0, x1, y1 = bbox
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), hex2bgr("#710193"), 2)
        return img

    def visualize_detection(self, img, bboxes_and_landmarks):
        img = np.copy(img)
        for b in bboxes_and_landmarks:
            # Display boxes and confidence scores
            cv2.putText(img,
                        f'{b[4]:.4f}',
                        (int(b[0]), int(b[1] + 12)),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.5,
                        (255, 255, 255))
            b = list(map(int,b))
            cv2.rectangle(img,
                          (b[0], b[1]),
                          (b[2], b[3]),
                          (0 , 0 ,255),2)
            
            # Display landmarks (face points)
            for i in range(5):
                cv2.circle(img,(b[5+i*2],b[6+i*2]),1,(0 , 0 ,255),4)
                
        return img

    def add_margin(self, bbox, left_margin_factor, right_margin_factor, top_margin_factor, bottom_margin_factor, img_width, img_height):
        x, y, w, h = bbox
        left = int(left_margin_factor * w)
        right = int(right_margin_factor * w)
        top = int(top_margin_factor * h)
        bottom = int(bottom_margin_factor * h)

        x = max(0, x - left)
        y = max(0, y - top)
        # w = min(img_width - x, w + left + right)
        # h = min(img_height - y, h + top + bottom)
        w = min(img_width, w + left + right)
        h = min(img_height, h + top + bottom)

        return int(x), int(y), int(w), int(h)

# Dictionary mapping node names to their corresponding classes
NODE_CLASS_MAPPINGS = {
    "FRED_CropFace": FRED_CropFace
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_CropFace": "👑 FRED_CropFace"
}
