# import gradio as gr
# import torch
# import faiss
# import numpy as np
# import pickle
# from transformers import CLIPModel, CLIPProcessor
# from PIL import Image
# import os

# DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
# INDEX_DIR = "index"

# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# model.eval()
# if DEVICE == "cuda":
#     model = model.half()

# index = faiss.read_index(os.path.join(INDEX_DIR, "image_index.faiss"))

# with open(os.path.join(INDEX_DIR, "image_paths.pkl"), "rb") as f:
#     image_paths = pickle.load(f)

# def search(query, top_k=12):
#     with torch.no_grad():
#         inputs = processor(text=[query], return_tensors="pt", padding=True).to(DEVICE)
#         if DEVICE == "cuda":
#             inputs = {k: v.half() if v.dtype == torch.float32 else v
#                       for k, v in inputs.items()}
#         text_emb = model.get_text_features(**inputs)
#         if hasattr(text_emb, 'pooler_output'):
#             text_emb = text_emb.pooler_output
#         text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
#         text_emb = text_emb.cpu().float().numpy().astype("float32")

#     scores, indices = index.search(text_emb, top_k)

#     results = []
#     for idx in indices[0]:
#         try:
#             img = Image.open(image_paths[idx]).convert("RGB")
#             results.append(img)
#         except:
#             pass
#     return results

# with gr.Blocks(title=" Image Search") as app:
#     gr.Markdown("# 🎥 Image Search")
#     with gr.Row():
#         query_box = gr.Textbox(placeholder="e.g. person with red jacket", label="Query")
#         top_k     = gr.Slider(4, 50, value=12, step=4, label="Results")
#         btn       = gr.Button("Search 🔍", variant="primary")
#     gallery = gr.Gallery(label="Results", columns=4, height=600)
#     btn.click(fn=search, inputs=[query_box, top_k], outputs=gallery)
#     query_box.submit(fn=search, inputs=[query_box, top_k], outputs=gallery)

# app.launch(server_port=8080)


import gradio as gr
import torch
import faiss
import numpy as np
import pickle
import json
import os
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from datetime import datetime

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
INDEX_DIR = "index"
OUTPUT_DIR = "output"  # folder where JSON files will be saved

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()
if DEVICE == "cuda":
    model = model.half()

index = faiss.read_index(os.path.join(INDEX_DIR, "image_index.faiss"))

with open(os.path.join(INDEX_DIR, "image_paths.pkl"), "rb") as f:
    image_paths = pickle.load(f)

def search(query, top_k=12):
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

    results = []
    images  = []

    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        path = image_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
        except:
            pass

        results.append({
            "rank":       rank + 1,
            "score":      round(float(score), 4),
            "image_path": path,
            "filename":   os.path.basename(path)
        })

    # ── Save JSON ──────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = "".join(c if c.isalnum() or c in " _-" else "_" for c in query).strip().replace(" ", "_")
    json_filename = f"{timestamp}_{safe_query}.json"
    json_path = os.path.join(OUTPUT_DIR, json_filename)

    json_data = {
        "query":        query,
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_results": len(results),
        "device":       DEVICE,
        "results":      results
    }

    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=4)

    print(f"✅ JSON saved: {json_path}")
    # ───────────────────────────────────────────────────────

    return images

with gr.Blocks(title="CCTV Image Search") as app:
    gr.Markdown("# 🎥 CCTV Image Search (GPU Powered)")
    gr.Markdown("Results are automatically saved as JSON in the `output/` folder.")
    with gr.Row():
        query_box = gr.Textbox(placeholder="e.g. person with red jacket", label="Query")
        top_k     = gr.Slider(4, 50, value=12, step=4, label="Results")
        btn       = gr.Button("Search 🔍", variant="primary")
    gallery = gr.Gallery(label="Results", columns=4, height=600)
    btn.click(fn=search, inputs=[query_box, top_k], outputs=gallery)
    query_box.submit(fn=search, inputs=[query_box, top_k], outputs=gallery)

app.launch(server_port=8080)