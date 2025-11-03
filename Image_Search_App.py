# image_search_app.py

import streamlit as st
import pickle
import search_engine
import numpy as np
import pillow_heif
from PIL import ImageOps
from PIL import Image
from pathlib import Path

# Adding comments to this portion to see if the alert system works

# Testing alert system for formatting changes

# Alert System is working!!!

try:
    pillow_heif.register_heif_opener()
except ImportError:
    print("pillow-heif not available — HEIC images may not load.")

st.set_page_config(layout="wide")

def correct_image_display(img):
    img = ImageOps.exif_transpose(img)
    return img

image_folder = Path(__file__).resolve().parent / "Images_Test"
base_path = Path(__file__).resolve().parent
embeddings_path = base_path / "face_db.pkl" 

with open(embeddings_path, "rb") as f:
    face_db = pickle.load(f)

# face_db is list of tuples: (name, embedding tensor)
known_names = [name for name, emb in face_db]
known_embeddings = [emb.numpy() if hasattr(emb, 'numpy') else emb for name, emb in face_db]

if isinstance(known_embeddings, list):
    known_embeddings = np.array(known_embeddings)

# Reshape from (N, 1, 512) → (N, 512)
if known_embeddings.ndim == 3 and known_embeddings.shape[1] == 1:
    known_embeddings = known_embeddings.reshape((known_embeddings.shape[0], -1))

st.title("Text + Face Image Search")

query_input = st.text_input("Enter your search query:", placeholder="Type a description or a person's name")
search_button = st.button("Search")

if "last_query" not in st.session_state:
    st.session_state["last_query"] = ""

if (query_input and (search_button or query_input != st.session_state["last_query"])):
    st.session_state["last_query"] = query_input
    query_lower = query_input.lower()
    # Check if any known face name is in query
    contains_face = any(name.lower() in query_lower for name in known_names)
    # Check if query has other words (for activity/scene)
    contains_scene = len(query_lower.split()) > 2 # more than just a face name
    print(f"contains_face: {contains_face}, contains_scene: {contains_scene}")


    # Decide which function to call
    if contains_face and contains_scene:
        all_results = search_engine.hybrid_search(
            query_input,
            known_embeddings,
            known_names,
            image_folder,
            alpha=0.5,
            top_k=12
        )
    else:
        all_results = search_engine.search_images(
            query_input,
            known_embeddings,
            known_names,
            image_folder
        )
    st.session_state["results"] = all_results[:12]

if "results" in st.session_state:
    results = st.session_state["results"]
    cols = st.columns(3)

    for i, result in enumerate(results[:12]):
        if isinstance(result[1], str):
            # Facial recognition result: (path, name, score)
            path, name, score = result
            caption = f"{Path(path).name} | Face Match: {score:.2f}"
        else:
            # Text-only CLIP result: (path, clip_score)
            path, score = result
            caption = f"CLIP Score: {score:.2f}"

        try:
            img = Image.open(path).convert("RGB")
            img = correct_image_display(img)

            with cols[i % 3]:
                st.image(img, caption=caption, use_column_width=True)
                with open(path, "rb") as img_file:
                    img_bytes = img_file.read()
                    st.download_button(
                        label="Download",
                        data=img_bytes,
                        file_name=Path(path).name,
                        mime="image/jpeg"
                    )
        except Exception as e:
            print(f"Error displaying image {path}: {e}")
