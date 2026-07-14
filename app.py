import streamlit as st
import pandas as pd
import numpy as np
import faiss
import torch
from transformers import CLIPModel, CLIPProcessor

# --- PAGE INITIALIZATION CONFIGURATION ---
st.set_page_config(
    page_title="Semantic Cross-Modal Retrieval Engine",
    page_icon="🔍",
    layout="wide"
)

# --- PROFESSIONAL DEVELOPER SIDEBAR CREDENTIALS ---
st.sidebar.markdown("### Developer Profile")
st.sidebar.markdown("**Sachin Kumar**")
st.sidebar.caption("B.Tech. in Mechanical Engg. & \nInterdisciplinary M.Tech. in Data Science")
st.sidebar.markdown("---")
st.sidebar.markdown("### System Architecture Topology")
st.sidebar.info(
    "Architecture separates concerns via standalone FastAPI endpoints querying structural vector indices "
    "cataloged by an offline PyTorch CLIP pipeline. Render requests hook onto asynchronous streams seamlessly."
)

st.title("🔍 Semantic Cross-Modal Image Retrieval Engine")
st.markdown("Enter a descriptive textual query to isolate matching conceptual imagery instantly using deep embedding matrix representations.")

# --- LAZY LOADING CACHED ASSETS ENGINE ---
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
    st.error(f"Initialization Error: Verify asset path layout structures. Details: {e}")
    st.stop()

# --- SEARCH PANEL USER INTERFACE ---
query_input = st.text_input("What sort of visual scene are you tracking?", placeholder="e.g., a white dog running on the beach")
top_k = st.slider("Select retrieval window size (Top K elements):", min_value=1, max_value=12, value=4)

if query_input:
    with st.spinner("Searching vector embedding spaces..."):
        try:
            # 1. Tokenize and preprocess input query string
            inputs = processor(text=[query_input], return_tensors="pt", padding=True)
            
            # 2. Extract text feature embedding vectors using CLIP
            with torch.no_grad():
                outputs = model.get_text_features(**inputs)
                
                if hasattr(outputs, "pooler_output"):
                    txt_embed = outputs.pooler_output.cpu().numpy()
                elif hasattr(outputs, "cpu"):
                    txt_embed = outputs.cpu().numpy()
                else:
                    txt_embed = np.array(outputs)
                
            # 3. Standard unit normalization for valid Inner Product matrix matching
            norm = np.linalg.norm(txt_embed, axis=1, keepdims=True)
            if norm > 0:
                txt_embed = txt_embed / norm
            txt_embed = txt_embed.astype('float32')
                
            # 4. Perform direct nearest-neighbor metric search on FAISS index
            distances, indices = image_index.search(txt_embed, top_k)
            
            # 5. Render results dynamically in responsive columns
            st.markdown(f"### Top {top_k} Structural Matches Found:")
            cols = st.columns(min(top_k, 4))
            
            for index, idx in enumerate(indices[0]):
                col_target = cols[index % 4]
                img_name = unique_images[idx]
                score = distances[0][index]
                
                # Fetch text captions from metadata to show below the image
                associated_captions = metadata[metadata['image'] == img_name]['caption'].tolist()
                primary_caption = associated_captions[0] if associated_captions else "No description cataloged."
                
                # Public open endpoint mirror on Hugging Face hosting the entire raw flickr8k photo collection
                public_hf_url = f"https://huggingface.co/datasets/kiddo/Flickr8k/resolve/main/Flicker8k_Dataset/{img_name}"
                
                with col_target:
                    try:
                        # Dynamically streams the actual photo directly from Hugging Face assets cache
                        st.image(public_hf_url, use_column_width=True, caption=f"Match #{index+1} (Score: {score:.4f})")
                        st.caption(f"📝 *\"{primary_caption}\"*")
                    except Exception:
                        # Fallback info card if a specific network connection drops out
                        st.info(f"📁 **File:** `{img_name}`\n\n🎯 Score: **{score:.4f}**\n\n📝 *\"{primary_caption}\"*")
                        
        except Exception as e:
            st.error(f"Processing anomaly detected: {e}")