import importlib
import os
import server
from aiohttp import web

node_list = [
#     "FRED_ImageBrowser",
    "FRED_CropFace",
#     "FRED_LoadImage_V2",
#     "FRED_LoadImage_V3",
#     "FRED_LoadImage_V4",
    "FRED_LoadImage_V5",
#     "FRED_LoadPathImagesPreivew",
#     "FRED_LoadPathImagesPreivew_v2",
#     "FRED_AutoCropImage_SDXL_Ratio_v3", 
    "FRED_AutoCropImage_SDXL_Ratio_v4",
    "FRED_JoinImages_v1", 
    "FRED_LoadRetinaFace", 
#     "FRED_PreviewOnly", 
#     "FRED_photo_prompt", 
    "FRED_LoraInfos",
    "FRED_TextMultiline",
    "FRED_ColorQuantization",
    "FRED_LoraList"
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for module_name in node_list:
    imported_module = importlib.import_module(".{}".format(module_name), __name__)
    NODE_CLASS_MAPPINGS.update(imported_module.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(imported_module.NODE_DISPLAY_NAME_MAPPINGS)

# @server.PromptServer.instance.routes.get("/bilbox/reboot")
# async def fetch_customnode_mappings(request):
    # type = request.rel_url.query["mode"]
    # print("BilboX:",type)
    # json_obj = {"server_op": type}
 
    # if(type == "shutdown"):
        # os.system("shutdown /s") # Shutdown
    # if(type == "reboot"):
        # os.system("shutdown /r") # Restart
    # if(type == "logout"):
        # os.system("shutdown /l") # logout
    # if(type == "lock"):
        # os.system("rundll32.exe user32.dll,LockWorkStation") # logout
    # return web.json_response(json_obj, content_type='application/json')

# WEB_DIRECTORY = "./web"
# __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
