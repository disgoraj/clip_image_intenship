import torch
import faiss
import numpy as np
import pickle
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import os
import time

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
INDEX_DIR = "index"

# Load model
print("Loading CLIP...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()
if DEVICE == "cuda":
    model = model.half()

# Load FAISS index on GPU
index = faiss.read_index(os.path.join(INDEX_DIR, "image_index.faiss"))

with open(os.path.join(INDEX_DIR, "image_paths.pkl"), "rb") as f:
    image_paths = pickle.load(f)

print(f"✅ Loaded {len(image_paths)} images in index")

def search(query, top_k=20):
    t = time.time()
    with torch.no_grad():
        inputs = processor(text=[query], return_tensors="pt", padding=True).to(DEVICE)
        if DEVICE == "cuda":
            inputs = {k: v.half() if v.dtype == torch.float32 else v
                      for k, v in inputs.items()}
        text_emb = model.get_text_features(**inputs)
        if hasattr(text_emb, 'pooler_output'):
            text_emb = text_emb.pooler_output
        text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
        text_emb = text_emb.cpu().float().numpy().astype("float32")

    scores, indices = index.search(text_emb, top_k)
    elapsed = (time.time() - t) * 1000

    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "path": image_paths[idx],
            "score": round(float(score), 4)
        })

    print(f"⚡ Search time: {elapsed:.1f}ms")
    return results

# Interactive CLI
while True:
    query = input("\n🔍 Query (or 'quit'): ").strip()
    if query.lower() == "quit":
        break
    results = search(query)
    for i, r in enumerate(results):
        print(f"  {i+1}. [{r['score']}] {r['path']}")