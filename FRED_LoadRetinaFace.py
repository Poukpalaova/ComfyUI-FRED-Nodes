import os
from facexlib.detection import RetinaFace
from .utils import models_dir

class FRED_LoadRetinaFace:
    models_dir = os.path.join(models_dir, 'facexlib')
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{}}
    
    RETURN_TYPES = ("RETINAFACE", )
    RETURN_NAMES = ("MODEL", )
    FUNCTION = "load"
    CATEGORY = "CFaceSwap"
    def load(self):
        from facexlib.detection import init_detection_model
        return (init_detection_model("retinaface_resnet50", model_rootpath=self.models_dir), )

# Dictionary mapping node names to their corresponding classes
NODE_CLASS_MAPPINGS = {
    "FRED_LoadRetinaFace": FRED_LoadRetinaFace
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_LoadRetinaFace": "👑 FRED_LoadRetinaFace"
}
