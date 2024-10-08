from __future__ import annotations

import gc
import numpy as np
import PIL.Image
import torch
from controlnet_aux.util import HWC3
from diffusers import (
    ControlNetModel,
    DiffusionPipeline,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

from cv_utils import resize_image
from preprocessor import Preprocessor
from settings import MAX_IMAGE_RESOLUTION, MAX_NUM_IMAGES

CANNY_MODEL_ID = "lllyasviel/control_v11p_sd15_canny"

def download_canny_controlnet_weight() -> None:
    ControlNetModel.from_pretrained(CANNY_MODEL_ID)

class Model:
    def __init__(self, base_model_id: str = "runwayml/stable-diffusion-v1-5"):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.base_model_id = base_model_id
        self.pipe = self.load_pipe(base_model_id)
        self.preprocessor = Preprocessor()

    def load_pipe(self, base_model_id: str) -> DiffusionPipeline:
        controlnet = ControlNetModel.from_pretrained(CANNY_MODEL_ID, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_model_id, safety_checker=None, controlnet=controlnet, torch_dtype=torch.float16
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        if self.device.type == "cuda":
            pipe.enable_xformers_memory_efficient_attention()
        pipe.to(self.device)
        torch.cuda.empty_cache()
        gc.collect()
        return pipe

    @torch.autocast("cuda")
    def run_pipe(
        self,
        prompt: str,
        negative_prompt: str,
        control_image: PIL.Image.Image,
        num_images: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> list[PIL.Image.Image]:
        generator = torch.Generator().manual_seed(seed)
        return self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            num_inference_steps=num_steps,
            generator=generator,
            image=control_image,
        ).images

    @torch.inference_mode()
    def process_canny(
        self,
        image: np.ndarray,
        prompt: str,
        additional_prompt: str,
        negative_prompt: str,
        num_images: int,
        image_resolution: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        low_threshold: int,
        high_threshold: int,
    ) -> list[PIL.Image.Image]:
        if image is None:
            raise ValueError("Input image cannot be None.")
        if image_resolution > MAX_IMAGE_RESOLUTION:
            raise ValueError("Image resolution exceeds the maximum allowed.")
        if num_images > MAX_NUM_IMAGES:
            raise ValueError("Number of images exceeds the maximum allowed.")

        self.preprocessor.load("Canny")
        control_image = self.preprocessor(
            image=image, low_threshold=low_threshold, high_threshold=high_threshold, detect_resolution=image_resolution
        )

        results = self.run_pipe(
            prompt=f"{prompt}, {additional_prompt}",
            negative_prompt=negative_prompt,
            control_image=control_image,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        return [control_image] + results
