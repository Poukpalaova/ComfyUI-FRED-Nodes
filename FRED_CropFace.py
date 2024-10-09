import torch
import numpy as np
import cv2  # Import OpenCV for face detection
from facexlib.detection import RetinaFace
from .utils import tensor2cv, cv2tensor, hex2bgr, BBox

class FRED_CropFace:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("RETINAFACE",),
                "image": ("IMAGE",),
                "confidence": ("FLOAT", {"default": 0.8, "min": 0, "max": 1}),
                "margin_factor": ("FLOAT", {"default": 1.0, "min": 0.0}),  # Margin as a factor of face width
                "face_id": ("INT", {"default": 0, "min": 0}),  # ID of the face to crop
            }
        }

    RETURN_TYPES = (
        "IMAGE", "IMAGE", "BBOX", "FLOAT"  # Added face pixel ratio
    )
    RETURN_NAMES = ("face_image", "preview", "bbox", "face_pixel_ratio")
    FUNCTION = "crop"
    CATEGORY = "FRED/image/postprocessing"

    def crop(self, model: RetinaFace, image: torch.Tensor, confidence: float, margin_factor: float, face_id: int):
        # Convert image to OpenCV format
        img_cv = tensor2cv(image)  # Function to convert Tensor to OpenCV image (BGR)
        img_height, img_width = img_cv.shape[:2]

        # Face detection
        with torch.no_grad():
            # bboxes: list of [x0, y0, x1, y1, confidence_score, five points (x, y)]
            bboxes = model.detect_faces(img_cv, confidence)

        if len(bboxes) == 0:
            print("WARNING! No face detected. Please adjust confidence or change picture. Input picture will be sent to output.")
            return image, image, np.zeros((4,)), 0.0  # Return the input image as output

        # Sort detected faces by x-coordinate (left to right)
        bboxes = sorted(bboxes, key=lambda b: b[0]) 

        # Handle face_id out of range
        if face_id >= len(bboxes):
            print(f"ERROR! Invalid face_id: {face_id}. Total detected faces: {len(bboxes)}. Using face_id = 0.")
            face_id = 0

        # Preview detected faces
        detection_preview = self.visualize_detection(img_cv, bboxes)

        # Add margin and make the box square
        bboxes = [
            self.add_margin_and_make_square(
                (int(min(x0, x1)), int(min(y0, y1)), int(abs(x1 - x0)), int(abs(y1 - y0))),
                margin_factor,
                img_width=img_width,
                img_height=img_height
            ) for (x0, y0, x1, y1, *_) in bboxes
        ]

        # Update preview with margins
        detection_preview = self.visualize_margin(detection_preview, bboxes)

        # Crop the selected face
        selected_bbox = bboxes[face_id]
        x, y, w, h = selected_bbox
        cropped_face = image[0, y:y + h, x:x + w, :].unsqueeze(0) 

        # Calculate the ratio of pixels occupied by the face
        face_pixels = w * h  # Area of the selected face
        total_pixels = img_height * img_width  # Total number of pixels in the image
        face_pixel_ratio = (face_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0

        return cropped_face, cv2tensor(detection_preview), selected_bbox, face_pixel_ratio
    
    def crop_faces(self, bboxes, image: torch.Tensor):
        """
        Returns: list of Tensor[h, w, c] of faces
        """
        return [image[0, y:y+h, x:x+w, :] for (x, y, w, h) in bboxes]

    def visualize_margin(self, img, bboxes):
        img = np.copy(img)
        for bbox in bboxes:
            x, y, w, h = bbox
            cv2.rectangle(img, (x, y), (x + w, y + h), hex2bgr("#710193"), 2)
        return img

    def visualize_detection(self, img, bboxes_and_landmarks):
        img = np.copy(img)
        for b in bboxes_and_landmarks:
            # Display boxes and confidence scores
            cv2.putText(img, f'{b[4]:.4f}', (int(b[0]), int(b[1] + 12)), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            b = list(map(int, b))
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            # Display landmarks (face points)
            for i in range(5):
                cv2.circle(img, (b[5 + i*2], b[6 + i*2]), 1, (0, 0, 255), 4)
        return img

    def add_margin_and_make_square(self, bbox, margin_factor: float, img_width: int, img_height: int):
        x, y, w, h = bbox
        margin_w, margin_h = int(margin_factor * w), int(margin_factor * h)

        # Add margin
        x = max(0, x - margin_w)
        y = max(0, y - margin_h)
        w = min(img_width - x, w + 2 * margin_w)
        h = min(img_height - y, h + 2 * margin_h)

        # Make the box square
        cx, cy = x + w // 2, y + h // 2
        max_side = max(w, h)
        x = max(0, cx - max_side // 2)
        y = max(0, cy - max_side // 2)
        w = h = min(max_side, img_width - x, img_height - y)
        
        return int(x), int(y), int(w), int(h)

# Dictionary mapping node names to their corresponding classes
NODE_CLASS_MAPPINGS = {
    "FRED_CropFace": FRED_CropFace
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_CropFace": "👑 FRED_CropFace"
}