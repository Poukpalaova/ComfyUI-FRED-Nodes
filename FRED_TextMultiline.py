import random

class FRED_TextMultiline:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "amount": ("INT", {"default": 1, "min": 1, "max": 8}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lines",)
    OUTPUT_IS_LIST = (True,)
    CATEGORY = "👑FRED/Util"
    FUNCTION = "select_multiline"

    def select_multiline(self, text: str, amount=1, seed=0):
        lines = text.strip().split("\n")
        lines = [line.strip() for line in lines if line.strip()]
        
        # Get starting index from seed
        start_index = seed % len(lines)
        
        # Get sequential lines starting from seed index
        selected_lines = []
        for i in range(amount):
            index = (start_index + i) % len(lines)
            selected_lines.append(lines[index])
            
        return (selected_lines,)

# Dictionary mapping node names to their corresponding classes
NODE_CLASS_MAPPINGS = {
    "FRED_TextMultiline": FRED_TextMultiline
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_TextMultiline": "👑 FRED_TextMultiline"
}