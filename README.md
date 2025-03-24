# Text-to-Image Generation with Stable Diffusion

This repository contains my personal project for learning and experimenting with **text-to-image generation using diffusion models**, specifically **Stable Diffusion v1.5**.

The goal was to understand the architecture, pipeline, and inference process in a fully offline setup. This project runs entirely on local hardware using PyTorch — no external API calls or internet needed once the weights are in place.

---

## Project Structure

```
StableDiffusion/
├── scripts/     # Model components: CLIP, DDPM, VAE, etc.
├── weights/     # Model weights (you must add manually)
├── output/      # Folder where generated images are saved
├── generate.py  # Interactive CLI entry point
├── requirements.txt  # Python dependencies
└── README.md    # You're reading it!
```

---

## Setup Instructions

### 1. 🔁 Clone the Repository

```bash
git clone https://github.com/Sayandip170900/Text-to-Image-Generation-with-Stable-Diffusion.git
cd StableDiffusion
```

### 2. (Optional) Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 📥 Download Required Weights

Weights are not included in this repository due to licensing restrictions. You must manually download and place them into the `weights/` directory.

#### ✅ Files Needed

| File | Description |
|------|-------------|
| vocab.json | CLIP tokenizer vocab |
| merges.txt | CLIP tokenizer merges |
| v1-5-pruned-emaonly.ckpt | Stable Diffusion model weights |

#### 🔗 Download From

- 🧠 **Hugging Face**: [Stable Diffusion v1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
  - Download `v1-5-pruned-emaonly.ckpt`
- 🧠 **Tokenizer Files**
  - Download `vocab.json` and `merges.txt`

Place all three files into the `weights/` folder in your project root.

---

## 🚀 Run the Generator

```bash
python generate.py
```

You will be prompted to enter:
- A text prompt (e.g., "a cat with sunglasses")
- An optional negative prompt (e.g., "blurry, low quality")

The generated image will be saved to:

```bash
output/generated.png
```

---

## Sample Outputs

### A cat with sunglasses  
![A cat with sunglasses](./output/A%20cat%20with%20sunglasses.jpg)

### A lion in the jungle  
![A lion in the jungle](./output/A%20lion%20in%20the%20jungle.jpg)

### A man fishing on the road, instead of a river  
![A man fishing on the road](./output/A%20man%20fishing%20on%20the%20road,%20instead%20of%20a%20river.jpg)

### Milky way galaxy as seen from a very clear sky  
![Milky way galaxy](./output/Milky%20way%20galaxy%20as%20seen%20from%20a%20very%20clear%20sky.jpg)

### Oil painting of a family on the beach  
![Oil painting](./output/Oil%20painting%20of%20a%20family%20on%20the%20beach.jpg)

### Spiderman in New York City  
![Spiderman](./output/Spiderman%20in%20New%20York%20City.jpg)

---

## Requirements

Python 3.10+ recommended

Install using:

```bash
pip install -r requirements.txt
```

**requirements.txt**
```txt
torch==2.6.0
transformers==4.47.1
Pillow==11.0.0
tqdm==4.67.1
safetensors
huggingface-hub
tokenizers
```

---

## Credits

- [**Stability AI**](https://stability.ai/) — for Stable Diffusion
- [**CompVis**](https://github.com/CompVis/stable-diffusion) — original repo
- [**Hugging Face**](https://huggingface.co/) — for hosting models and tokenizer files
- [**Umar Jamil**](https://github.com/hkproj) — for explanation and guidance
