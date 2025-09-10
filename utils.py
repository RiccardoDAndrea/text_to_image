import os
import requests
from diffusers import AutoPipelineForText2Image, DEISMultistepScheduler
import torch
import nvidia_smi


def load_lottieurl(url: str):
    """Lädt eine Lottie-Animation von einer URL (JSON)."""
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"[WARN] Konnte Lottie-Animation nicht laden: {e}")
    return None

def get_gpu_memory():
    nvidia_smi.nvmlInit()
    try:
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        mem_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        total = mem_info.total / 1024**3  # in GB
        used = mem_info.used / 1024**3   # in GB
        free = mem_info.free / 1024**3   # in GB
    finally:
        nvidia_smi.nvmlShutdown()
    return total, used, free
