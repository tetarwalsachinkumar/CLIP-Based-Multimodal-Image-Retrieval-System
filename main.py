from fastapi import FastAPI, Query
import pandas as pd
import numpy as np
import faiss
import torch
from transformers import CLIPModel, CLIPProcessor

app = FastAPI(title="CLIP Semantic Vector Search API")

# Load pipeline models and datasets instantly into cache on boot
device = "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
image_index = faiss.read_index("data/clip_embed.image_index.faiss")
metadata = pd.read_csv("data/flickr8k_metadata.csv")
unique_images = metadata['image'].unique()

@app.get("/search")
def search_images(q: str = Query(..., description="Text query string"), k: int = 4):
    try:
        # Vectorize natural language query
        inputs = processor(text=[q], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model.get_text_features(**inputs)
            txt_embed = outputs.cpu().numpy()
            
        # Standard unit normalization for structural cosine similarity matching
        norm = np.linalg.norm(txt_embed, axis=1, keepdims=True)
        if norm > 0:
            txt_embed = txt_embed / norm
            
        # Search the FAISS database index
        distances, indices = image_index.search(txt_embed.astype('float32'), k)
        
        # Build systematic JSON matches output array
        matches = []
        for rank in range(k):
            idx = indices[0][rank]
            matches.append({
                "rank": rank + 1,
                "filename": unique_images[idx],
                "score": float(distances[0][rank])
            })
        return {"status": "success", "query": q, "results": matches}
    except Exception as e:
        return {"status": "error", "message": str(e)}