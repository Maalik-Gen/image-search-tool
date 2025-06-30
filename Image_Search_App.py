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

# Orientation fix for UI only
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

# Load all pickle files
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

# Load or initialize feedback data
if "feedback_data" not in st.session_state:
    try:
        with open("feedback_data.pkl", "rb") as f:
            st.session_state["feedback_data"] = pickle.load(f)
    except FileNotFoundError:
        st.session_state["feedback_data"] = {}

feedback_data = st.session_state["feedback_data"]

# Save feedback to disk
def save_feedback():
    with open("feedback_data.pkl", "wb") as f:
        pickle.dump(st.session_state["feedback_data"], f)

# CLIP + Feedback search
def search_images_streamlit(query, top_k=10, similarity_threshold=0.2):
    text_tokens = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_embedding = clip_model.encode_text(text_tokens)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

    results = []
    for filename, image_emb in clip_embeddings.items():
        similarity = torch.cosine_similarity(text_embedding, image_emb).item()

        # Apply feedback bonus
        bonus = 0.0
        if query in feedback_data and filename in feedback_data[query]:
            score = feedback_data[query][filename]
            bonus = 0.2 * score  # Boost weight

        adjusted_score = similarity + bonus
        if adjusted_score > similarity_threshold:
            results.append((filename, adjusted_score))

    results.sort(key=lambda x: x[1], reverse=True)

    final = []
    for filename, score in results[:top_k]:
        path = image_folder / filename
        try:
            img = Image.open(path).convert("RGB")
            img = correct_image_display(img)
            final.append((img, filename, score))
        except Exception as e:
            print(f"Error loading {path}: {e}")
    return final

# UI
st.markdown("## Text-Based Image Search + Feedback")

query_input = st.text_input("Search Prompt", placeholder="e.g. People playing tennis")
search_button = st.button("Search")

# Run search if button clicked or prompt changes
if query_input:
    if search_button or st.session_state.get("last_query") != query_input:
        st.session_state["last_query"] = query_input
        st.session_state["results"] = search_images_streamlit(query_input)

# Display results
if "results" in st.session_state and st.session_state["results"]:
    cols = st.columns(3)
    for i, (img, filename, score) in enumerate(st.session_state["results"]):
        with cols[i % 3]:
            st.image(img, caption=f"{filename} ({score:.2f})", use_container_width=True)

            # Thumbs up feedback
            if st.button("👍", key=f"up_{i}"):
                feedback_data.setdefault(query_input, {})
                feedback_data[query_input][filename] = feedback_data[query_input].get(filename, 0) + 1
                save_feedback()
                st.session_state["results"] = search_images_streamlit(query_input)
                st.rerun()

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
