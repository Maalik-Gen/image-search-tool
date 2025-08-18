import os
import cv2
import torch
import clip
import pickle
import numpy as np
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from gfpgan import GFPGANer


# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-L/14@336px", device=device)
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
face_detector = YOLO("yolov11l-face.pt")

face_restore = GFPGANer(
    model_path="GFPGANv1.4.pth",
    upscale=1,
    arch="clean",
    channel_multiplier=2,
    bg_upsampler=None
)

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


def get_face_embedding(face_pil):
    face_tensor = transform(face_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = facenet(face_tensor)
    embedding_np = embedding.squeeze().cpu().numpy()
    return embedding_np / np.linalg.norm(embedding_np)


def match_face(embedding, known_embeddings, threshold=0.7):
    similarities = cosine_similarity(embedding.reshape(1, -1), known_embeddings)[0]
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]
    if best_score >= threshold:
        return best_idx, best_score
    return None, None


def load_search_faces(search_faces_path):
    with open(search_faces_path, "rb") as f:
        data = pickle.load(f)  # list of (image_relative_path, embedding tensor)
    image_paths = [entry[0] for entry in data]
    embeddings = np.array([entry[1].numpy() if hasattr(entry[1], 'numpy') else entry[1] for entry in data])
    return image_paths, embeddings


def is_person_name(query, known_names):
    return any(name.lower() in query.lower() for name in known_names)


def split_name_and_context(query, known_names):
    """Find a known name in query and return (name, remaining_text)"""
    for name in known_names:
        if name.lower() in query.lower():
            remaining = query.lower().replace(name.lower(), "").strip()
            return name, remaining if remaining else None
    return None, query

def search_images(query, known_embeddings, known_names, image_folder):
    image_paths = list(Path(image_folder).rglob("*.[jp][pn]g"))

    # Load face search database (pre-extracted embeddings for all images)
    search_face_paths, search_face_embeddings = load_search_faces("search_face_db.pkl")

    # Face-only search (check if query is a known name)
    if query.lower() in [n.lower() for n in known_names]:
        print("This is the Search Image Face Function")
        matched_face_images = []
        for idx, search_emb in enumerate(search_face_embeddings):
            match_idx, match_score = match_face(search_emb, known_embeddings)
            if match_idx is not None and known_names[match_idx].lower() == query.lower():
                matched_face_images.append((Path(image_folder) / search_face_paths[idx], match_score))

        matched_face_images.sort(key=lambda x: x[1], reverse=True)
        return matched_face_images[:12]

    # CLIP-only search (precomputed embeddings assumed loaded)
    with torch.no_grad():
        print("This is the Search Image Clip Function")
        text_embedding = clip_model.encode_text(
            clip.tokenize([query]).to(device)
        ).cpu().numpy()[0]

    results = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            img_input = clip_preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                img_embedding = clip_model.encode_image(img_input).cpu().numpy()[0]
            score = np.dot(text_embedding, img_embedding) / (
                np.linalg.norm(text_embedding) * np.linalg.norm(img_embedding)
            )
            results.append((path, score))
        except:
            continue

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:12]

def dynamic_alpha(context_text):
    word_count = len(context_text.split())
    if word_count >= 4:
        return 0.8  # Strong scene/context influence
    elif word_count == 1:
        return 0.6  # Still some context, but not huge
    else:
        return 0.7  # Medium context


    
def hybrid_search(query_input, known_embeddings, known_names, image_folder, alpha=0.5, top_k=12):
    print("Hybrid search called")
    
    person_name, clip_text = split_name_and_context(query_input, known_names)
    if person_name is None:
        print("No person name found in query")
        return []
    
    with open("clip_embeddings.pkl", "rb") as f:
        clip_dict = pickle.load(f)  
    
    search_face_paths, search_face_embeddings = load_search_faces("search_face_db.pkl")
    
    matched_face_paths = []
    for idx, search_emb in enumerate(search_face_embeddings):
        match_idx, match_score = match_face(search_emb, known_embeddings)
        if match_idx is not None and known_names[match_idx].lower() == person_name.lower():
            matched_face_paths.append(search_face_paths[idx])  # relative paths
    
    # Find images that have both face and clip embeddings
    matched_paths_set = set(matched_face_paths)
    clip_paths_set = set(clip_dict.keys())
    common_paths = matched_paths_set & clip_paths_set
    
    if not common_paths:
        print("No common images with face and clip embeddings found")
        return []
    
    # Embed query text
    with torch.no_grad():
        text_emb = clip_model.encode_text(clip.tokenize([clip_text]).to(device)).cpu().numpy()[0]
    
    combined_scores = {}
    for rel_path in common_paths:
        clip_emb = clip_dict[rel_path].cpu().numpy().squeeze()
        clip_score = np.dot(text_emb, clip_emb) / (np.linalg.norm(text_emb) * np.linalg.norm(clip_emb))
        face_score = 1.0  # matched face is binary 1.0
        alpha = dynamic_alpha(clip_text) 
        combined_scores[rel_path] = alpha * clip_score + (1 - alpha) * face_score
    
    ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    # Return full paths with scores
    return [(image_folder / Path(rel_path), score) for rel_path, score in ranked]






