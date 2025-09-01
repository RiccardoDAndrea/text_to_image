### Text to image model with Huggingface 

# -- Importing Dependencies
import os, configparser
from huggingface_hub import InferenceClient

def read_configuration_file(path=str):
    """
    Import API Key for Huggingface
    ---

    path : str
        Location of the Credentials File

    Return 
    ----
    Api Key
    """
    parser = configparser.ConfigParser()
    parser.read("credentials.conf")

    Huggingface_API_Key = parser.get("API_Key", "Huggingface_API_Key")

    return Huggingface_API_Key

Huggingface_API_Key = read_configuration_file()

# Create Client with Model (provider) and the API Key
client = InferenceClient(
    provider="fal-ai",
    api_key=Huggingface_API_Key)

# Description to create a image
image = client.text_to_image(
    """
    A desolate, abandoned kindergarten with a single, 
    rocking crib casting a long, distorted shadow. The wallpaper peels off, revealing a hidden, 
    grotesque drawing of a child with gouged-out eyes. The air shimmers in an unnatural, cold haze.
    """,
    model="black-forest-labs/FLUX.1-dev")

image.show()           # Öffnet das Bild
image.save("bild.png") # Optional: Speichern