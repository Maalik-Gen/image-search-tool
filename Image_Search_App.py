import streamlit as st
import torch
import clip
import pickle
import io
from PIL import Image, ExifTags
from pathlib import Path

# HEIC support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    print("pillow-heif not available — HEIC images may not load.")

# Fix orientation from EXIF
def fix_image_orientation(img):
    try:
        for tag in ExifTags.TAGS.keys():
            if ExifTags.TAGS[tag] == 'Orientation':
                orientation_tag = tag
                break
        exif = img._getexif()
        if exif is not None:
            orientation = exif.get(orientation_tag)
            if orientation == 3:
                img = img.rotate(180, expand=True)
            elif orientation == 6:
                img = img.rotate(270, expand=True)
            elif orientation == 8:
                img = img.rotate(90, expand=True)
    except:
        pass
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

# CLIP search
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
            img = fix_image_orientation(img)
            final.append((img, f"{filename} ({score:.2f})"))
        except Exception as e:
            print(f"Error loading {path}: {e}")
    return final

# Streamlit UI
st.set_page_config(layout="wide")
st.markdown("Text-Based Image Search with CLIP")

query_input = st.text_input("Search Prompt", placeholder="e.g. people at dinner")
search_button = st.button("Search")

# Trigger search if Enter was pressed (text changed) or button was clicked
if query_input and (search_button or st.session_state.get("last_query") != query_input):
    st.session_state["last_query"] = query_input  # Prevent rerun loop

    results_display = search_images_streamlit(query_input)

    if results_display:
        cols = st.columns(3)
        for i, (img, caption) in enumerate(results_display):
            filename = caption.split(" (")[0]  # Extract filename if needed
            with cols[i % 3]:
                st.image(img, caption=caption, use_container_width=True)

                # Convert image to bytes
                img_bytes = io.BytesIO()
                img.save(img_bytes, format="PNG")
                img_bytes.seek(0)

                # Add download button
                st.download_button(
                    label="Download",
                    data=img_bytes,
                    file_name=filename.split("/")[-1],
                    mime="image/png"
                )
