import nvidia_smi

nvidia_smi.nvmlInit()

handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)  # Erste GPU
mem_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

total_vram = mem_info.total / 1024**2   # in MB
used_vram = mem_info.used / 1024**2     # in MB
free_vram = mem_info.free / 1024**2     # in MB

print(f"Gesamt VRAM: {total_vram:.2f} MB")
print(f"Genutztes VRAM: {used_vram:.2f} MB")
print(f"Freies VRAM: {free_vram:.2f} MB")

nvidia_smi.nvmlShutdown()
