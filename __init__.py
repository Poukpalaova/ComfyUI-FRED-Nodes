import importlib
import os
import server
from aiohttp import web

# WEB_DIRECTORY = "./web"

node_list = [
#     "FRED_ImageBrowser",
    "FRED_CropFace",
#     "FRED_LoadImage_V2",
#     "FRED_LoadImage_V3",
#     "FRED_LoadImage_V4",
    "FRED_LoadImage_V5",
    "FRED_LoadImage_V6",
    "FRED_LoadImage_V7",
    "FRED_LoadImage_V8",
#     "FRED_LoadPathImagesPreivew",
#     "FRED_LoadPathImagesPreivew_v2",
#     "FRED_AutoCropImage_SDXL_Ratio_v3", 
    "FRED_AutoCropImage_SDXL_Ratio_v4",
    "FRED_AutoCropImage_Native_Ratio_v5",
    "FRED_JoinImages_v1", 
#     "FRED_LoadRetinaFace", 
#     "FRED_PreviewOnly", 
#     "FRED_photo_prompt", 
    "FRED_LoraInfos",
    "FRED_TextMultiline",
#     "FRED_ColorQuantization",
#     "FRED_LoraList",
    "FRED_Text_to_XMP",
    "FRED_AutoImageTile_from_Mask_v1",
    "FRED_Simplified_Parameters_Panel",
    "FRED_ImageQualityInspector",
    "FRED_ImageUncropFromBBox"
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for module_name in node_list:
    imported_module = importlib.import_module(".{}".format(module_name), __name__)
    NODE_CLASS_MAPPINGS.update(imported_module.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(imported_module.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
