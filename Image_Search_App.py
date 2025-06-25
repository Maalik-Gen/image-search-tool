import streamlit as st
import torch
import clip
import pickle
import io
from PIL import Image
from pathlib import Path

# Set Streamlit page config
st.set_page_config(layout="wide")

# HEIC support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    print("pillow-heif not available — HEIC images may not load.")


# New heuristic display fix: force portrait layout in UI
def correct_image_display(img):
    try:
        if img.width > img.height:
            img = img.rotate(90, expand=True)
    except Exception as e:
        print(f"Orientation fallback failed: {e}")
    return img


# Setup
device = "cpu"
clip_model, preprocess_clip = clip.load("ViT-B/32", device=device)

@st.cache_resource
def load_pickles():
    with open("clip_embeddings.pkl", "rb") as f:
        clip_embeddings = pickle.load(f)
    with open("face_db.pkl", "rb") as f:
        face_db = pickle.load(f)
    clip_embeddings = {k: v.to(device) for k, v in clip_embeddings.items()}
    face_db = [(k, v.to(device)) for k, v in face_db]
    return clip_embeddings, face_db

clip_embeddings, face_db = load_pickles()
image_folder = Path(__file__).resolve().parent / "Images_Test"


# CLIP search function
def search_images_streamlit(query, top_k=10, similarity_threshold=0.2):
    text_tokens = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_embedding = clip_model.encode_text(text_tokens)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

    results = []
    for filename, image_emb in clip_embeddings.items():
        similarity = torch.cosine_similarity(text_embedding, image_emb).item()
        if similarity > similarity_threshold:
            results.append((filename, similarity))

    results.sort(key=lambda x: x[1], reverse=True)

    final = []
    for filename, score in results[:top_k]:
        path = image_folder / filename
        try:
            img = Image.open(path).convert("RGB")
            img = correct_image_display(img)
            final.append((img, f"{filename} ({score:.2f})"))
        except Exception as e:
            print(f"Error loading {path}: {e}")
    return final


# UI
st.markdown("## Text-Based Image Search with CLIP")

query_input = st.text_input("Search Prompt", placeholder="e.g. people at dinner")
search_button = st.button("Search")

# Run search either by button or Enter key
if query_input and (search_button or st.session_state.get("last_query") != query_input):
    st.session_state["last_query"] = query_input
    results_display = search_images_streamlit(query_input)

    if results_display:
        cols = st.columns(3)
        for i, (img, caption) in enumerate(results_display):
            filename = caption.split(" (")[0]
            with cols[i % 3]:
                st.image(img, caption=caption, use_container_width=True)

                # Download button
                img_bytes = io.BytesIO()
                img.save(img_bytes, format="PNG")
                img_bytes.seek(0)

                st.download_button(
                    label="Download",
                    data=img_bytes,
                    file_name=filename.split("/")[-1],
                    mime="image/png"
                )
