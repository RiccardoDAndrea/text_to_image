from diffusers import StableDiffusionPipeline
import torch

def create_pipe(model_path: str):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to("cuda")

    return pipe


def prompt_and_negative_prompt(prompt: str, negative_prompt: str):
    return prompt, negative_prompt


def generateImage():
    pipe = create_pipe(
        model_path="/run/media/riccardodandrea/Ricca_Data/hub/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
    )
    prompt, negative_prompt = prompt_and_negative_prompt(
        prompt="""(Moonlit lighting:1.1) A close-up of a mysterious man's face, 
               with soft shadows highlighting his sharp features, the light reflecting 
               in his intense eyes, cool tones creating a calm yet enigmatic atmosphere""",
        negative_prompt="fake eyes, deformed eyes, bad eyes, cgi, 3D, digital, airbrushed"
    )

    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        height=512,   # SD 1.5 arbeitet am besten mit 512x512
        width=512
    ).images[0]

    return image


Image = generateImage()
Image.save("Picture/juggernaut_test.png")
print("✅ Bild gespeichert: juggernaut_test.png")
