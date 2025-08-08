import torch
import clip
from pathlib import Path
from PIL import Image
import pickle
import numpy as np
from facenet_pytorch import InceptionResnetV1
from ultralytics import YOLO
from tqdm import tqdm

print("Embedding generation")

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

device = "cpu"
base_path = Path(__file__).resolve().parent
image_folder = base_path / "Images_Test"
known_faces_path = base_path / "known_faces"
valid_exts = [".jpg", ".jpeg", ".png", ".heic", ".heif"]

clip_model, preprocess_clip = clip.load("ViT-B/16", device=device)
facenet_model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
yolo_model = YOLO("yolov11l-face.pt")

def get_face_crop(img):
    results = yolo_model(img)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            if conf > 0.5:
                face = img.crop((x1, y1, x2, y2)).resize((160, 160))
                return face
    return None

# CLIP embeddings
image_embeddings = {}
image_paths = [p for p in image_folder.rglob("*") if p.suffix.lower() in valid_exts and not p.name.startswith("._")]

for path in image_paths:
    try:
        img = Image.open(path).convert("RGB")
        img_tensor = preprocess_clip(img).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = clip_model.encode_image(img_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        image_embeddings[str(path.relative_to(image_folder))] = embedding.cpu()
    except Exception:
        pass

with open("clip_embeddings.pkl", "wb") as f:
    pickle.dump(image_embeddings, f)

# FaceNet embeddings
face_db = []
known_face_paths = [p for p in known_faces_path.rglob("*") if p.suffix.lower() in valid_exts and not p.name.startswith("._")]

for path in tqdm(known_face_paths, desc="FaceNet embeddings"):
    try:
        img = Image.open(path).convert("RGB")
        face_crop = get_face_crop(img)
        if face_crop is not None:
            face_tensor = torch.tensor(np.array(face_crop)).permute(2, 0, 1).float() / 255.0
            face_tensor = face_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = facenet_model(face_tensor)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            person_name = path.stem.lower().replace("_", " ").strip()
            face_db.append((person_name, embedding.cpu()))
    except Exception:
        pass

with open("face_db.pkl", "wb") as f:
    pickle.dump(face_db, f)

# FaceNet embeddings for all images in Images_Test (to support face search)
search_face_db = []

for path in tqdm(image_paths, desc="FaceNet embeddings for Images_Test"):
    try:
        img = Image.open(path).convert("RGB")
        face_crop = get_face_crop(img)
        if face_crop is not None:
            face_tensor = torch.tensor(np.array(face_crop)).permute(2, 0, 1).float() / 255.0
            face_tensor = face_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = facenet_model(face_tensor)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            search_face_db.append((str(path.relative_to(image_folder)), embedding.cpu()))
    except Exception:
        pass

with open("search_face_db.pkl", "wb") as f:
    pickle.dump(search_face_db, f)

print(f"Saved {len(search_face_db)} face embeddings for Images_Test.")
