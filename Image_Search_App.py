import streamlit as st
import torch
import clip
import pickle
import io
from PIL import Image
from pathlib import Path
from facenet_pytorch import MTCNN

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
mtcnn = MTCNN(image_size=160, margin=0, device=device)

# Load all pickles
@st.cache_resource
def load_pickles():
    with open("clip_embeddings.pkl", "rb") as f:
        clip_embeddings = pickle.load(f)
    with open("face_db.pkl", "rb") as f:
        face_db = pickle.load(f)
    try:
        with open("feedback_data.pkl", "rb") as f:
            feedback_data = pickle.load(f)
    except FileNotFoundError:
        feedback_data = {}
    clip_embeddings = {k: v.to(device) for k, v in clip_embeddings.items()}
    face_db = [(k, v.to(device)) for k, v in face_db]
    return clip_embeddings, face_db, feedback_data

clip_embeddings, face_db, feedback_data = load_pickles()
image_folder = Path(__file__).resolve().parent / "Images_Test"

# Save feedback scores to disk
def save_feedback():
    with open("feedback_data.pkl", "wb") as f:
        pickle.dump(feedback_data, f)

# Save feedback face crop to separate file
def save_feedback_face(query, filename, img):
    face = mtcnn(img)
    if face is not None:
        feedback_face_db_path = Path("feedback_face_db.pkl")
        try:
            if feedback_face_db_path.exists():
                with open(feedback_face_db_path, "rb") as f:
                    face_entries = pickle.load(f)
            else:
                face_entries = {}
            face_entries.setdefault(query, [])
            face_entries[query].append((filename, face.cpu()))
            with open(feedback_face_db_path, "wb") as f:
                pickle.dump(face_entries, f)
            print(f"Saved face for feedback: {filename}")
        except Exception as e:
            print(f"Failed to save feedback face: {e}")

# CLIP + Feedback search
def search_images_streamlit(query, top_k=10, similarity_threshold=0.2):
    text_tokens = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_embedding = clip_model.encode_text(text_tokens)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

    results = []
    for filename, image_emb in clip_embeddings.items():
        similarity = torch.cosine_similarity(text_embedding, image_emb).item()

        bonus = 0.0
        score = 0.0
        if query in feedback_data and filename in feedback_data[query]:
            score = feedback_data[query][filename]
            score = max(-2, min(5, score))  # Clamp score between -2 and 5
            bonus = 0.2 * score
            print(f"[BONUS] {filename} — score: {score} → bonus: {bonus:.2f}")

        adjusted_score = similarity + bonus
        if adjusted_score > similarity_threshold:
            results.append((filename, adjusted_score, similarity, bonus, score))

    results.sort(key=lambda x: x[1], reverse=True)

    final = []
    for filename, score, raw_score, bonus, feedback_score in results[:top_k]:
        path = image_folder / filename
        try:
            img = Image.open(path).convert("RGB")
            img = correct_image_display(img)
            final.append((img, filename, score, feedback_score, bonus))
        except Exception as e:
            print(f"Error loading {path}: {e}")
    return final

# UI
st.markdown("## Text-Based Image Search + Feedback")

query_input = st.text_input("Search Prompt", placeholder="e.g. people at dinner")
search_button = st.button("Search")

# NEW: allow search on Enter or Search button
if "last_query" not in st.session_state:
    st.session_state["last_query"] = ""

if (query_input and (search_button or query_input != st.session_state["last_query"])):
    st.session_state["last_query"] = query_input
    st.session_state["results_display"] = search_images_streamlit(query_input)

results_display = st.session_state.get("results_display", [])


# Results display
if results_display:
    cols = st.columns(3)
    for i, (img, filename, score, feedback_score, bonus) in enumerate(results_display):
        with cols[i % 3]:
            st.image(img, caption=f"{filename} ({score:.2f})", use_container_width=True)
            st.caption(f"Feedback score: {feedback_score} | Bonus: {bonus:.2f}")

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("👍", key=f"up_{i}"):
                    feedback_data.setdefault(query_input, {})
                    feedback_data[query_input][filename] = feedback_data[query_input].get(filename, 0) + 1
                    save_feedback()
                    save_feedback_face(query_input, filename, img)
                    st.rerun()

            with col2:
                if st.button("👎", key=f"down_{i}"):
                    feedback_data.setdefault(query_input, {})
                    current = feedback_data[query_input].get(filename, 0)
                    feedback_data[query_input][filename] = current - 0.5
                    save_feedback()
                    st.rerun()

            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG")
            img_bytes.seek(0)
            st.download_button(
                label="Download",
                data=img_bytes,
                file_name=filename.split("/")[-1],
                mime="image/png"
            )
