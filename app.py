import streamlit as st
import pandas as pd
import numpy as np
import faiss
import torch
import os
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

st.set_page_config(
    page_title="Semantic Cross-Modal Retrieval Engine",
    page_icon="🔍",
    layout="wide"
)

# --- SIDEBAR DETAILS ---
st.sidebar.markdown("### Developer Profile")
st.sidebar.markdown("**Sachin Kumar**")
st.sidebar.caption("B.Tech. in Mechanical Engg. & \nInterdisciplinary M.Tech. in Data Science")
st.sidebar.markdown("---")
st.sidebar.markdown("### System Infrastructure")
st.sidebar.info(
    "This application implements a decoupled conceptual model matching system using OpenAI's CLIP "
    "and optimized FAISS matrix arrays."
)

st.title("🔍 Semantic Cross-Modal Image Retrieval Engine")

@st.cache_resource
def load_assets():
    device = "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    image_index = faiss.read_index("data/clip_embed.image_index.faiss")
    metadata = pd.read_csv("data/flickr8k_metadata.csv")
    return model, processor, image_index, metadata

try:
    model, processor, image_index, metadata = load_assets()
    unique_images = metadata['image'].unique()
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

query_input = st.text_input("What sort of visual scene are you tracking?", placeholder="e.g., a white dog running")
top_k = st.slider("Select retrieval window size (Top K elements):", 1, 12, 4)

if query_input:
    with st.spinner("Searching vector spaces..."):
        inputs = processor(text=[query_input], return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model.get_text_features(**inputs)
            txt_embed = outputs.cpu().numpy()
            
        norm = np.linalg.norm(txt_embed, axis=1, keepdims=True)
        if norm > 0:
            txt_embed = txt_embed / norm
            
        distances, indices = image_index.search(txt_embed.astype('float32'), top_k)
        
        st.markdown("### Top Matches Found:")
        cols = st.columns(min(top_k, 4))
        
        for index, idx in enumerate(indices[0]):
            col_target = cols[index % 4]
            img_name = unique_images[idx]
            score = distances[0][index]
            local_path = f"data/Images/{img_name}"
            
            with col_target:
                if os.path.exists(local_path):
                    st.image(Image.open(local_path), use_column_width=True, caption=f"Score: {score:.4f}")
                else:
                    st.warning(f"Vector verified: {img_name}")
                    st.caption(f"Score: {score:.4f}")