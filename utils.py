import os
import requests
from diffusers import AutoPipelineForText2Image, DEISMultistepScheduler
import torch


def load_lottieurl(url: str):
    """Lädt eine Lottie-Animation von einer URL (JSON)."""
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"[WARN] Konnte Lottie-Animation nicht laden: {e}")
    return None


