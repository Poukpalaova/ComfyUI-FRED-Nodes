from server import PromptServer
from nodes import PreviewImage
import torch

class FRED_PreviewOnly(PreviewImage):
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "func"
    CATEGORY = "👑FRED/image"
    INPUT_IS_LIST = True
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {"images": ("IMAGE",)},
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "id": "UNIQUE_ID"},
        }

    def func(self, id, **kwargs):
        # images_in = torch.cat(images)
        images_in = torch.cat(kwargs.pop('images'))

        id = id[0]

        # Ensure that extra_pnginfo is in the correct format
        extra_pnginfo = kwargs.get('extra_pnginfo', [{}])

        if isinstance(extra_pnginfo, list) and extra_pnginfo and isinstance(extra_pnginfo[0], dict):
            extra_pnginfo = extra_pnginfo[0]

        expected_kwargs = {
            'prompt': kwargs.get('prompt', None),
            'extra_pnginfo': extra_pnginfo
        }

        print(f"images_in type: {type(images_in)}, content: {images_in}")
        print(f"extra_pnginfo type: {type(extra_pnginfo)}, content: {extra_pnginfo}")

        # call PreviewImage base
        ret = self.save_images(images=images_in, **kwargs)

        # send the images to view
        PromptServer.instance.send_sync("early-image-handler", {"id": id, "urls":ret['ui']['images']})

        return (images,)

# Dictionary mapping node names to their corresponding classes
NODE_CLASS_MAPPINGS = {
    "FRED_PreviewOnly": FRED_PreviewOnly
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_PreviewOnly": "👑 FRED_PreviewOnly"
}