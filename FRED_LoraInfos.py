import folder_paths
import json
import os
import server
from aiohttp import web

# @server.PromptServer.instance.routes.post('/lora_info')
# async def fetch_lora_info(request):
    # post = await request.post()
    # lora_name = post.get("lora_name")
    # (output, triggerWords, examplePrompt, baseModel) = get_lora_info(lora_name)

    # return web.json_response({"output": output, "triggerWords": triggerWords, "examplePrompt": examplePrompt, "baseModel": baseModel})

class FRED_LoraInfos:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        # Récupère la liste des fichiers LORA disponibles
        LORA_LIST = sorted(folder_paths.get_filename_list("loras"), key=str.lower)
        return {
            "required": {
                "lora_name": (LORA_LIST,)
            }
        }

    RETURN_NAMES = ("lora_name", "triggerWords", "examplePrompt")
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    FUNCTION = "lora_info"
    OUTPUT_NODE = True
    CATEGORY = "FRED/lora"

    # def get_json_info(self, lora_name):
        # lora_path = folder_paths.get_full_path("loras", lora_name)
        # json_path = os.path.splitext(lora_path)[0] + '.json'
        
        # try:
            # with open(json_path, 'r', encoding='utf-8') as f:
                # data = json.load(f)
                # baseModel = data.get('baseModel', '')
                # triggerWords = data.get('triggerWords', '')
                # if isinstance(triggerWords, list):
                    # triggerWords = ', '.join(triggerWords)
                
                # # Keep examplePrompt as a list
                # examplePrompt = data.get('examplePrompt', [])
                
                # # Format the output display with line breaks
                # output = f"Trigger Words: {triggerWords}\n"
                # if examplePrompt:
                    # output += "Example Prompts:\n"
                    # output += '\n'.join(str(prompt) for prompt in examplePrompt)
                
                # # Return examplePrompt as a list for compatibility with select_multiline
                # return output, triggerWords, examplePrompt, baseModel
        # except FileNotFoundError:
            # return "No JSON file found", "", [], ""
        # except json.JSONDecodeError:
            # return "Invalid JSON file", "", [], ""
    def get_json_info(self, lora_name):
        lora_path = folder_paths.get_full_path("loras", lora_name)
        json_path = os.path.splitext(lora_path)[0] + '.json'
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                baseModel = data.get('baseModel', '')
                triggerWords = data.get('triggerWords', '')
                if isinstance(triggerWords, list):
                    triggerWords = ', '.join(triggerWords)
                
                # Get examplePrompt as a string with newlines
                examplePrompt = data.get('examplePrompt', [])
                if isinstance(examplePrompt, list):
                    examplePrompt = '\n'.join(str(prompt) for prompt in examplePrompt)
                
                output = f"Trigger Words: {triggerWords}\n"
                if examplePrompt:
                    output += f"Example Prompts:\n{examplePrompt}"
                
                return output, triggerWords, examplePrompt, baseModel
        except FileNotFoundError:
            return "No JSON file found", "", "", ""
        except json.JSONDecodeError:
            return "Invalid JSON file", "", "", ""
    def lora_info(self, lora_name):
        output, triggerWords, examplePrompt, baseModel = self.get_json_info(lora_name)
        # return {"ui": {"text": (output,)}, "result": (lora_name, triggerWords, examplePrompt)}
        return {"ui": {"text": (output,), "model": (baseModel,)}, "result": (lora_name, triggerWords, examplePrompt)}

# Dictionary mapping node names to their corresponding classes
NODE_CLASS_MAPPINGS = {
    "FRED_LoraInfos": FRED_LoraInfos
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_LoraInfos": "👑 FRED_LoraInfos"
}