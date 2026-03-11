import os
import glob
import torch
import numpy as np
import faiss
import pickle
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import time

# ── Config ──────────────────────────────────────────
IMAGES_DIR    = "images"
INDEX_DIR     = "index"
BATCH_SIZE    = 256       # increase if VRAM allows (e.g. 512 for 24GB GPU)
NUM_WORKERS   = 8         # parallel image loading threads
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME    = "openai/clip-vit-base-patch32"
# ────────────────────────────────────────────────────

print(f"✅ Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

os.makedirs(INDEX_DIR, exist_ok=True)

# Load CLIP model on GPU
print("\nLoading CLIP model...")
model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()

# Enable half precision for 2x speed on GPU
if DEVICE == "cuda":
    model = model.half()  # float16
    print("✅ Using float16 (half precision) for speed")

# Collect all image paths
image_paths = []
for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"]:
    image_paths += glob.glob(os.path.join(IMAGES_DIR, "**", ext), recursive=True)

print(f"\n📁 Found {len(image_paths)} images")

# Fast parallel image loader
def load_image(path):
    try:
        return Image.open(path).convert("RGB")
    except:
        return None

def load_batch_parallel(paths):
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        images = list(executor.map(load_image, paths))
    valid = [(img, p) for img, p in zip(images, paths) if img is not None]
    return zip(*valid) if valid else ([], [])

# Encode all images
all_embeddings = []
valid_paths = []
start = time.time()

with torch.no_grad():
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="🔄 Encoding"):
        batch_paths = image_paths[i:i + BATCH_SIZE]
        images, paths = load_batch_parallel(batch_paths)

        if not images:
            continue

        inputs = processor(
            images=list(images),
            return_tensors="pt",
            padding=True
        ).to(DEVICE)

        # Convert to half precision
        if DEVICE == "cuda":
            inputs = {k: v.half() if v.dtype == torch.float32 else v
                      for k, v in inputs.items()}

        embeddings = model.get_image_features(**inputs)
        if hasattr(embeddings, 'pooler_output'):
            embeddings = embeddings.pooler_output
        embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
        all_embeddings.append(embeddings.cpu().float().numpy())
        valid_paths.extend(paths)

elapsed = time.time() - start
print(f"\n⚡ Encoded {len(valid_paths)} images in {elapsed:.1f}s")
print(f"⚡ Speed: {len(valid_paths)/elapsed:.0f} images/sec")

# Stack embeddings
all_embeddings = np.vstack(all_embeddings).astype("float32")

# Build FAISS GPU Index
print("\nBuilding FAISS index...")
dimension = all_embeddings.shape[1]  # 512

index = faiss.IndexFlatIP(dimension)
index.add(all_embeddings)
faiss.write_index(index, os.path.join(INDEX_DIR, "image_index.faiss"))

with open(os.path.join(INDEX_DIR, "image_paths.pkl"), "wb") as f:
    pickle.dump(valid_paths, f)

print(f"✅ Done! Indexed {len(valid_paths)} images")
print(f"💾 Index saved to {INDEX_DIR}/")