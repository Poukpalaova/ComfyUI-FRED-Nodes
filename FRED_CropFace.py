import torch
import numpy as np
import cv2
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
                "margin_factor": ("FLOAT", {"default": 1.0, "min": 0.0}),
                "face_id": ("INT", {"default": 0, "min": 0}),
                "max_size": ("INT", {"default": 1536, "min": 256}), # Maximum size for detection
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "BBOX", "FLOAT")
    RETURN_NAMES = ("face_image", "preview", "bbox", "face_pixel_ratio")
    FUNCTION = "crop"
    CATEGORY = "FRED/image/postprocessing"

    def crop(self, model: RetinaFace, image: torch.Tensor, confidence: float, margin_factor: float, face_id: int, max_size: int):
        img_cv = tensor2cv(image)
        img_height, img_width = img_cv.shape[:2]

        # Resize image for face detection if it's too large
        scale = 1
        if max(img_width, img_height) > max_size:
            scale = max_size / max(img_width, img_height)
            img_resized = cv2.resize(img_cv, (int(img_width * scale), int(img_height * scale)))
        else:
            img_resized = img_cv

        # Face detection on resized image
        with torch.no_grad():
            bboxes = model.detect_faces(img_resized, confidence)

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

        # Add margin and make square on original image size
        bboxes = [
            self.add_margin_and_make_square(
                (int(min(x0, x1)), int(min(y0, y1)), int(abs(x1 - x0)), int(abs(y1 - y0))),
                margin_factor,
                img_width=img_width,
                img_height=img_height
            ) for (x0, y0, x1, y1, *_) in bboxes
        ]

        detection_preview = self.visualize_margin(detection_preview, bboxes)

        # Crop the selected face from the original image
        selected_bbox = bboxes[face_id]
        x, y, w, h = selected_bbox
        cropped_face = image[0, y:y + h, x:x + w, :].unsqueeze(0)
        
        face_pixels = w * h
        total_pixels = img_height * img_width
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
