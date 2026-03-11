# clip_image_intenship
# 🔍 CLIP Image Search Engine
### *Search your local images using natural language — fully offline, GPU-accelerated*

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2011.8-red?style=for-the-badge&logo=pytorch)
![CLIP](https://img.shields.io/badge/OpenAI-CLIP-green?style=for-the-badge)
![FAISS](https://img.shields.io/badge/Meta-FAISS-blue?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**Type `apple in hand` → instantly get all matching images from your local collection.**  
Like Google Lens, but 100% offline and running on YOUR machine.

</div>

---

## ✨ What Is This?

This project lets you **search through thousands of local images using plain English text** — powered by OpenAI's CLIP model and Meta's FAISS similarity search engine.

No internet needed. No API keys. No cloud. Everything runs locally on your GPU.

> **Examples of queries that work:**
> - `"apple"` → finds all apple photos
> - `"apple in hand"` → finds specifically that
> - `"sunset over mountains"` → finds scenic landscape shots
> - `"person wearing red shirt"` → finds people in red
> - `"cat sleeping on sofa"` → you get the idea

---

## 🚀 Features

- 🧠 **AI-powered semantic search** — understands meaning, not just filenames
- ⚡ **GPU-accelerated** — CUDA 11.8 support for blazing fast encoding & search
- 🔍 **Smart relevance filtering** — only shows genuinely related images, no junk results
- 🖼️ **Image-to-image search** — upload a photo to find visually similar ones (Google Lens style)
- 💾 **JSON output** — every search auto-saves results to `./results/` folder
- 🌐 **Beautiful Web UI** — built with Gradio, runs in your browser
- 📴 **100% Offline** — works with zero internet after first model download
- 🗂️ **Scales well** — tested with thousands of images, sub-second search

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Vision-Language Model | OpenAI CLIP (`ViT-B/32`) via HuggingFace Transformers |
| Similarity Search | Meta FAISS (GPU-accelerated) |
| Deep Learning | PyTorch + CUDA 11.8 |
| Web UI | Gradio |
| Language | Python 3.10 |

---

## 📋 Requirements

- **OS:** Windows 10/11 (64-bit)
- **GPU:** NVIDIA GTX 1650 / 1660 / 1070 / 1080 or better
- **CUDA:** 11.8
- **Python:** 3.10.x
- **RAM:** 8GB+ recommended
- **Disk:** ~3GB (model + dependencies)

---

## ⚙️ Installation

### 1. Install CUDA 11.8
Download from NVIDIA: https://developer.nvidia.com/cuda-11-8-0-download-archive  
Select: `Windows → x86_64 → 11 → exe (local)`

### 2. Install Python 3.10
Download from: https://www.python.org/downloads/release/python-31011/  
⚠️ **Check "Add Python to PATH"** during installation!

### 3. Clone this repository
```bash
git clone https://github.com/yourusername/clip-image-search.git
cd clip-image-search
```

### 4. Create virtual environment
```bash
py -3.10 -m venv venv
venv\Scripts\activate
```

### 5. Install dependencies
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch with CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install FAISS GPU
pip install faiss-gpu

# Install remaining dependencies
pip install transformers Pillow numpy tqdm gradio
```

### 6. Verify GPU is detected
```bash
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0)); print('CUDA:', torch.cuda.is_available())"
```
Expected output:
```
GPU: NVIDIA GeForce GTX 1650
CUDA: True
```

---

## 📁 Project Structure

```
clip-image-search/
│
├── images/              ← 📸 Put ALL your images here
│   ├── apple.jpg
│   ├── sunset.png
│   └── ...
│
├── index/               ← 🗂️ Auto-created: FAISS index files
│   ├── images.faiss
│   └── image_paths.pkl
│
├── results/             ← 💾 Auto-created: JSON search results saved here
│   ├── apple.json
│   └── sunset over mountains.json
│
├── build_index.py       ← 🔨 Run ONCE to index your images
├── search.py            ← 🔍 Core search engine + CLI mode
└── app.py               ← 🌐 Web UI (Gradio)
```

---

## 🚀 Usage

### Step 1 — Add your images
Put all your images (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`) into the `./images/` folder.

### Step 2 — Build the index (run once)
```bash
python build_index.py
```
This encodes all your images using CLIP and builds a FAISS search index.  
Only needs to run **once** (or when you add new images).

### Step 3a — Launch Web UI
```bash
python app.py
```
Then open your browser at: **http://localhost:7860**

### Step 3b — Use CLI
```bash
python search.py
```
```
🔍 Query (or 'quit'): apple in hand

Rank   Score    Filename
--------------------------------------------------
  1    0.2950   apple_in_hand.jpg
  2    0.2710   fruit_basket.jpg

✅ 2 related image(s) found
💾 JSON saved → results/apple in hand.json
```

---

## 📄 JSON Output Format

Every search automatically saves a `.json` file to `./results/`:

```json
{
  "query": "apple in hand",
  "total_results": 2,
  "threshold": 0.20,
  "results": [
    {
      "rank": 1,
      "path": "images/apple_in_hand.jpg",
      "filename": "apple_in_hand.jpg",
      "score": 0.2950
    },
    {
      "rank": 2,
      "path": "images/fruit_basket.jpg",
      "filename": "fruit_basket.jpg",
      "score": 0.2710
    }
  ]
}
```

---

## ⚡ Performance

| Images | Index Build Time (GPU) | Search Time |
|--------|----------------------|-------------|
| 1,000 | ~30 seconds | <20ms |
| 10,000 | ~5 minutes | <50ms |
| 100,000 | ~45 minutes | <100ms |

*Tested on GTX 1650 4GB VRAM*

---

## 🎛️ Configuration

You can tune these settings in `search.py`:

```python
# Minimum score to consider an image "related"
# 0.20 = loosely related | 0.25 = clearly related | 0.30 = very similar
RELEVANCE_THRESHOLD = 0.20
```

And in `build_index.py`:
```python
BATCH_SIZE = 64    # Increase if you have more VRAM (e.g. 128 for 8GB VRAM)
MODEL_NAME = "ViT-B/32"   # Change to "ViT-L/14" for better accuracy (needs more VRAM)
```

---

## 🔧 Troubleshooting

| Problem | Fix |
|---------|-----|
| `CUDA not available` | Reinstall PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu118` |
| `H:\ drive not found` | Add `os.environ["HF_HOME"] = "C:\\clip_cache"` at top of file |
| `No images found` | Make sure images are inside `./images/` folder |
| `Score -340282...` | Update to latest code — old bug, now fixed |
| Out of VRAM | Reduce `BATCH_SIZE` in `build_index.py` |
| Web UI won't open | Use `http://localhost:7860` not `http://0.0.0.0:7860` |

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

1. Fork the repo
2. Create your branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- [OpenAI CLIP](https://github.com/openai/CLIP) — Vision-language model
- [HuggingFace Transformers](https://huggingface.co/openai/clip-vit-base-patch32) — Model hosting
- [Meta FAISS](https://github.com/facebookresearch/faiss) — Similarity search
- [Gradio](https://gradio.app/) — Web UI framework

---

<div align="center">
  <b>Built with ❤️ | Fully Offline | GPU Powered</b>
</div>
