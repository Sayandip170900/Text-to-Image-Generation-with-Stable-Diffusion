from scripts.pipeline import generate
from scripts.model_loader import preload_models_from_standard_weights
from transformers import CLIPTokenizer
from PIL import Image
import torch
import os
import subprocess
import platform
import re

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = CLIPTokenizer("weights/vocab.json", merges_file="weights/merges.txt")
models = preload_models_from_standard_weights("weights/v1-5-pruned-emaonly.ckpt", device=DEVICE)
os.makedirs("output", exist_ok=True)

def filename(prompt: str) -> str:
    safe_prompt = re.sub(r'[^\w\s-]', '', prompt)
    return safe_prompt.strip().replace(' ', '_') or "generated"

while True:
    print("\nEnter your prompt (type 'X' to quit):")
    prompt = input("> ").strip()
    if prompt.lower() == "x":
        print("Exiting.")
        break
    if not prompt:
        print("Empty prompt. Skipping.")
        continue

    uncond_prompt = input("Enter a negative prompt (optional): ").strip()

    print("Generating image...")
    output_image = generate(
        prompt=prompt,
        uncond_prompt=uncond_prompt,
        do_cfg=True,
        cfg_scale=7.5,
        sampler_name="ddpm",
        n_inference_steps=50,
        seed=42,
        models=models,
        device=DEVICE,
        idle_device="cpu",
        tokenizer=tokenizer,
    )

    safe_filename = filename(prompt) + ".jpg"
    output_path = os.path.join("output", safe_filename)
    Image.fromarray(output_image).save(output_path)
    print(f"Image saved to: {output_path}")

    try:
        if platform.system() == "Windows":
            os.startfile(output_path)
        elif platform.system() == "Darwin":
            subprocess.run(["open", output_path])
        else:
            subprocess.run(["feh", output_path])
    except Exception as e:
        print(f"Could not open image viewer: {e}")
