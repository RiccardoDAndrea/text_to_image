import os
# set Hugging Face cache dir
os.environ["HF_HOME"] = "/run/media/riccardodandrea/Ricca_Data/hub"
from diffusers import AutoPipelineForText2Image, DEISMultistepScheduler, StableDiffusionPipeline

import torch

pipe = pipe = StableDiffusionPipeline.from_pretrained("UnfilteredAI/NSFW-gen-v2", 
                                                   torch_dtype=torch.float16,)
                                                       
pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = """
A Wither with a party hat that is much too small for his medium-sized head, desperately trying to hang a garland of Creeper heads. He is having trouble controlling the explosive decoration, and one of the Creeper heads has already detonated, blowing a hole in his right upper arm. In the background, happy sheep can be seen wearing party hats and eating cupcakes with blue frosting.
"""

generator = torch.manual_seed(33)
image = pipe(prompt, generator=generator, num_inference_steps=25).images[0]  
image.save("Picture/Iron.png")