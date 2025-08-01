from __future__ import annotations
import comfy.samplers
import sys
from comfy.comfy_types.node_typing import ComfyNodeABC, InputTypeDict, IO

class FRED_Simplified_Parameters_Panel(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "model": ("MODEL",),
                "scale": (IO.FLOAT, {"default": 1.4, "min": 0.1, "max": 10.0, "step": 0.01}),
                "denoise": (IO.FLOAT, {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "guidance": (IO.FLOAT, {"default": 2.2, "min": 0.0, "max": 30.0, "step": 0.1}),
                "steps": (IO.INT, {"default": 8, "min": 1, "max": 200, "control_after_generate": True}),
                "noise_seed": (IO.INT, {"default": 42, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "sampler": (comfy.samplers.SAMPLER_NAMES, ),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES, ),
            }
        }

    RETURN_TYPES = (
        IO.FLOAT,  # scale
        IO.FLOAT,  # denoise
        IO.FLOAT,  # guidance
        IO.INT,    # steps
        IO.INT,    # seed
        "SAMPLER", # sampler
        "STRING",       # sampler name
        "SIGMAS",  # scheduler
        "STRING",       # scheduler name
    )

    RETURN_NAMES = (
        "scale",
        "denoise",
        "guidance",
        "steps",
        "noise",
        "sampler",
        "sampler_name",
        "sigmas",
        "scheduler_name",
    )

    FUNCTION = "execute"
    CATEGORY = "👑FRED/Util"

    def execute(self, model, scale: float, denoise: float, guidance: float, steps: int, noise_seed: int, sampler: str, scheduler: str) -> tuple[float, float, float, int, int, str, str]:
        sampler_name = sampler
        scheduler_name = scheduler
        sampler = comfy.samplers.sampler_object(sampler)

        total_steps = steps
        if denoise < 1.0:
            if denoise <= 0.0:
                return (torch.FloatTensor([]),)
            total_steps = int(steps/denoise)

        sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, total_steps).cpu()
        sigmas = sigmas[-(steps + 1):]
        return (
            scale,
            denoise,
            guidance,
            steps,
            Noise_RandomNoise(noise_seed),
            sampler,
            sampler_name,
            sigmas,
            scheduler_name,
        )

# Dictionary mapping node names to their corresponding classes
NODE_CLASS_MAPPINGS = {
    "FRED_Simplified_Parameters_Panel": FRED_Simplified_Parameters_Panel
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FRED_Simplified_Parameters_Panel": "👑 FRED Simplified Parameters Panel"
}