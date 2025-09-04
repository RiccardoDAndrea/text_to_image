from diffusers import StableDiffusionXLPipeline
import torch

def create_pipe(model_path:str):
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )

    pipe = pipe.to("cuda")
    return pipe

# --- Prompt
def prompt_and_negative_prompt(prompt:str, negative_prompt:str):
    return prompt, negative_prompt


# --- Bild erzeugen

def generateImage():
    pipe = create_pipe(model_path="/run/media/riccardodandrea/Ricca_Data/hub/models--UnfilteredAI--NSFW-gen-v2/snapshots/982782a450570e5f064016b404d4b7a1c19dbad5")

    prompt, negative_prompt = prompt_and_negative_prompt(
                                prompt="""
                                A car with 5 weehls""",
                                                         
                                negative_prompt = "fake eyes, deformed eyes, bad eyes, cgi, 3D, digital, airbrushed""")
    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        height=1024,
        width=1024
        ).images[0]

    return image

Image = generateImage()
# --- Speichern
Image.save("Picture/juggernaut_test.png")
print("✅ Bild gespeichert: juggernaut_test.png")
