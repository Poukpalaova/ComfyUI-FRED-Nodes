import nodes

class FRED_Advanced_multi_parameters_panel_v1:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_scale": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 4.0, "step": 0.1}),
                "denoise_global": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "guidance_global": ("FLOAT", {"default": 2.2, "min": 0.0, "max": 10.0, "step": 0.1}),
                "steps_global": ("INT", {"default": 8, "min": 1, "max": 50, "step": 1}),
                "activate_zone_1": ("BOOLEAN", {"default": True}),
                "denoise_zone_1": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "guidance_zone_1": ("FLOAT", {"default": 2.2, "min": 0.0, "max": 10.0, "step": 0.1}),
                "steps_zone_1": ("INT", {"default": 8, "min": 1, "max": 50, "step": 1}),
                "activate_zone_2": ("BOOLEAN", {"default": True}),
                "denoise_zone_2": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "guidance_zone_2": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "steps_zone_2": ("INT", {"default": 6, "min": 1, "max": 50, "step": 1}),
                "activate_zone_3": ("BOOLEAN", {"default": True}),
                "denoise_zone_3": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "guidance_zone_3": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "steps_zone_3": ("INT", {"default": 10, "min": 1, "max": 50, "step": 1}),
                "activate_upscale": ("BOOLEAN", {"default": True}),
                "upscale_denoise": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "upscale_guidance": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "upscale_steps": ("INT", {"default": 10, "min": 1, "max": 50, "step": 1}),
                "upscale_auto_grid": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = (
        "FLOAT",  # image_scale
        "FLOAT", "FLOAT", "INT",  # global: denoise, guidance, steps
        "BOOLEAN", "FLOAT", "FLOAT", "INT",  # zone1: activate, denoise, guidance, steps
        "BOOLEAN", "FLOAT", "FLOAT", "INT",  # zone2
        "BOOLEAN", "FLOAT", "FLOAT", "INT",  # zone3
        "BOOLEAN", "FLOAT", "FLOAT", "INT", "BOOLEAN"   # upscale: denoise, guidance, steps, auto_grid
    )

    RETURN_NAMES = (
        "image_scale",
        "denoise_global", "guidance_global", "steps_global",
        "activate_zone_1", "denoise_zone_1", "guidance_zone_1", "steps_zone_1",
        "activate_zone_2", "denoise_zone_2", "guidance_zone_2", "steps_zone_2",
        "activate_zone_3", "denoise_zone_3", "guidance_zone_3", "steps_zone_3",
        "activate_upscale", "upscale_denoise", "upscale_guidance", "upscale_steps", "upscale_auto_grid"
    )

    FUNCTION = "control"
    CATEGORY = "👑FRED/Util"

    def control(self, image_scale,
                denoise_global, guidance_global, steps_global,
                activate_zone_1, denoise_zone_1, guidance_zone_1, steps_zone_1,
                activate_zone_2, denoise_zone_2, guidance_zone_2, steps_zone_2,
                activate_zone_3, denoise_zone_3, guidance_zone_3, steps_zone_3,
                activate_upscale, upscale_denoise, upscale_guidance, upscale_steps, upscale_auto_grid):
        return (
            image_scale,
            denoise_global, guidance_global, steps_global,
            activate_zone_1, denoise_zone_1, guidance_zone_1, steps_zone_1,
            activate_zone_2, denoise_zone_2, guidance_zone_2, steps_zone_2,
            activate_zone_3, denoise_zone_3, guidance_zone_3, steps_zone_3,
            activate_upscale, upscale_denoise, upscale_guidance, upscale_steps, upscale_auto_grid
        )
        
# Dictionary mapping node names to their corresponding classes
NODE_CLASS_MAPPINGS = {
    "FRED_Advanced_multi_parameters_panel_v1": FRED_Advanced_multi_parameters_panel_v1
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_Advanced_multi_parameters_panel_v1": "👑 FRED_Advanced_multi_parameters_panel_v1"
}