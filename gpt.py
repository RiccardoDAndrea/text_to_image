
import os
os.environ["HF_HOME"] = "/run/media/riccardodandrea/Ricca_Data"
import torch
from transformers import pipeline
model_id = "EleutherAI/gpt-neo-2.7B"


pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
]

outputs = pipe(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])
