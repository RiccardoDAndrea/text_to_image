import os
# set Hugging Face cache dir
os.environ["HF_HOME"] = "/run/media/riccardodandrea/Ricca_Data"
from diffusers import AutoPipelineForText2Image, DEISMultistepScheduler
import torch

pipe = AutoPipelineForText2Image.from_pretrained('UnfilteredAI/NSFW-gen-v2', torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = """
A portrait of a muscular, anxious young man wearing a mech suit. The light falls from 
the top left. Light bokeh, intricate, steel metal, elegant, sharp focus, soft lighting, 
vibrant colors. The mech armor iron man stylish. 
"""

generator = torch.manual_seed(33)
image = pipe(prompt, generator=generator, num_inference_steps=25).images[0]  
image.save("Picture/dreamshaper.png")
