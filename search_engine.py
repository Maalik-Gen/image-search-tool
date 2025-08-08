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
clip_model, clip_preprocess = clip.load("ViT-B/16", device=device)
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


def search_images(query, known_embeddings, known_names, image_folder):
    # Load all image paths
    image_paths = list(Path(image_folder).rglob("*.[jp][pn]g"))
    image_embeddings = []

    for path in image_paths:
        if not path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            continue
        if path.name.startswith("._"):
            continue
        try:
            image = Image.open(path).convert("RGB")
        except UnidentifiedImageError:
            print(f"Skipping unreadable image: {path}")
            continue

        image_input = clip_preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = clip_model.encode_image(image_input)

        image_embeddings.append((path, embedding.cpu().numpy()[0]))

    # Encode query text with CLIP
    with torch.no_grad():
        text_embedding = clip_model.encode_text(clip.tokenize([query]).to(device)).cpu().numpy()[0]

    results = []

    # Load face embeddings
    search_face_paths, search_face_embeddings = load_search_faces("search_face_db.pkl")

    if is_person_name(query, known_names):
        final_matches = []

        # Step 1: Facial match
        for idx, search_emb in enumerate(search_face_embeddings):
            match_idx, match_score = match_face(search_emb, known_embeddings)
            if match_idx is not None:
                matched_name = known_names[match_idx]
                if matched_name.lower() in query.lower():
                    image_path = Path(image_folder) / search_face_paths[idx]
                    final_matches.append((image_path, matched_name, match_score))

        # Step 2: Filter CLIP scores to only face-matched images
        clip_scores = []
        for img_path, name, face_score in final_matches:
            try:
                img = Image.open(img_path).convert("RGB")
                img_input = clip_preprocess(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    img_embedding = clip_model.encode_image(img_input).cpu().numpy()[0]
                clip_score = np.dot(text_embedding, img_embedding) / (np.linalg.norm(text_embedding) * np.linalg.norm(img_embedding))
                clip_scores.append((img_path, name, clip_score))
            except Exception as e:
                print(f"Skipping {img_path} due to error: {e}")

        # Step 3: Sort by CLIP score (relevance to full query)
        clip_scores.sort(key=lambda x: x[2], reverse=True)
        return clip_scores[:12]

    else:
        # CLIP-only results
        for path, img_embed in image_embeddings:
            score = np.dot(text_embedding, img_embed) / (np.linalg.norm(text_embedding) * np.linalg.norm(img_embed))
            results.append((path, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:12]

