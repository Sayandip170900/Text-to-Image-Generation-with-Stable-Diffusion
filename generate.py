from scripts.pipeline import generate
from scripts.model_loader import preload_models_from_standard_weights
from transformers import CLIPTokenizer
from PIL import Image
import torch
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

prompt = input("Enter your prompt: ").strip()
if not prompt:
    print("No prompt entered. Exiting.")
    exit(1)

uncond_prompt = input("Enter a negative prompt (optional): ").strip()

tokenizer = CLIPTokenizer("weights/vocab.json", merges_file="weights/merges.txt")
models = preload_models_from_standard_weights("weights/v1-5-pruned-emaonly.ckpt", device=DEVICE)

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

os.makedirs("output", exist_ok=True)
output_path = os.path.join("output", "generated.jpg")
Image.fromarray(output_image).save(output_path)
print(f"Image saved to: {output_path}")
