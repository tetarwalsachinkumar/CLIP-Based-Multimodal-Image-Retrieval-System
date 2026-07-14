import streamlit as st
import requests
import os
from PIL import Image

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
st.sidebar.markdown("### System Infrastructure Topology")
st.sidebar.info(
    "Architecture separates concern via standalone FastAPI endpoints querying structural vector indices "
    "cataloged by an offline PyTorch CLIP pipeline. Render requests hook onto asynchronous streams seamlessly."
)

st.title("🔍 Semantic Cross-Modal Image Retrieval Engine")
st.markdown("Enter a descriptive textual query to isolate matching conceptual imagery instantly using deep embedding matrix representations.")

query_input = st.text_input("What sort of visual scene are you tracking?", placeholder="e.g., a white dog running on the beach")
top_k = st.slider("Select retrieval window size (Top K elements):", min_value=1, max_value=12, value=4)

if query_input:
    with st.spinner("Communicating with FastAPI vector backend services..."):
        try:
            # Send HTTP requests directly to the backend engine API endpoint
            response = requests.get(f"http://127.0.0.1:8000/search?q={query_input}&k={top_k}")
            
            if response.status_code == 200:
                data = response.json()
                if data["status"] == "success":
                    st.markdown(f"### Top {top_k} Structural Vectors Retrieved:")
                    cols = st.columns(min(top_k, 4))
                    
                    for index, match in enumerate(data["results"]):
                        col_target = cols[index % 4]
                        img_name = match["filename"]
                        score = match["score"]
                        
                        # Set up path pointing to your local subset folder
                        local_path = f"data/Images/{img_name}"
                        
                        with col_target:
                            if os.path.exists(local_path):
                                st.image(Image.open(local_path), use_column_width=True, caption=f"Match #{index+1} (Score: {score:.4f})")
                            else:
                                st.warning(f"Vector verified: {img_name}")
                                st.caption(f"Score: {score:.4f}")
            else:
                st.error("Backend REST framework communication interruption.")
        except Exception as e:
            st.error(f"Failed to query local server endpoint: {e}")