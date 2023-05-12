from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil
import torch

# stage 1
stage_1 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
#stage_1.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
stage_1.enable_model_cpu_offload()
stage_1.enable_attention_slicing(1)

prompt= "Circuit board with text that says Floyd IF"
prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt)
image = stage_1(prompt_embeds=prompt_embeds, negative_prompt_embeds= negative_embeds, num_inference_steps=15)
imagepng=image.images[0]
imagepng.save("RTX 2060.png")

safety_modules = {
    "feature_extractor": stage_1.feature_extractor,
    "safety_checker": stage_1.safety_checker,
    "watermarker": stage_1.watermarker,
}

stage_3 = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler", **safety_modules, torch_dtype=torch.float16
)
stage_3.enable_model_cpu_offload()
stage_3.enable_attention_slicing(1)

stage_3 = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler", **safety_modules, torch_dtype=torch.float16)

HDimg = stage_3(prompt=prompt, image=imagepng).images[0]
HDimg.save("RTXHD.png")