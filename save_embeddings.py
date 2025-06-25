import torch
import clip
from pathlib import Path
from PIL import Image
import pickle
from facenet_pytorch import MTCNN, InceptionResnetV1
from tqdm import tqdm

# HEIC support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    print("pillow-heif not available — HEIC images may not load.")

# Setup
device = "cpu"
base_path = Path(__file__).resolve().parent
image_folder = base_path / "Images_Test"
known_faces_path = base_path / "known_faces"
valid_exts = [".jpg", ".jpeg", ".png", ".heic", ".heif"]

# Load models
clip_model, preprocess_clip = clip.load("ViT-B/32", device=device)
mtcnn = MTCNN(image_size=160, margin=0, device=device)
facenet_model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# FaceNet embeddings for known faces
face_db = []
for path in known_faces_path.rglob("*"):
    if path.suffix.lower() not in valid_exts or path.name.startswith("._"):
        continue
    try:
        img = Image.open(path).convert("RGB")
        face_tensor = mtcnn(img)
        if face_tensor is not None:
            face_tensor = face_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = facenet_model(face_tensor)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            face_db.append((str(path), embedding.cpu()))
    except Exception as e:
        print(f"Face error: {path}: {e}")

with open("face_db.pkl", "wb") as f:
    pickle.dump(face_db, f)

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
    except Exception as e:
        print(f"CLIP error: {path}: {e}")

with open("clip_embeddings.pkl", "wb") as f:
    pickle.dump(image_embeddings, f)

print("Saved FaceNet and CLIP embeddings.")

# Face matching (store known matches only)
results = {}
FACE_MATCH_THRESHOLD = 0.8

print("Matching test images with known faces...")
for image_path in tqdm(image_paths):
    try:
        img = Image.open(image_path).convert("RGB")
        face_tensor = mtcnn(img)
        if face_tensor is None:
            continue
        face_tensor = face_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            test_embedding = facenet_model(face_tensor)

        best_match = None
        best_dist = float("inf")

        for known_path, known_emb in face_db:
            dist = torch.norm(test_embedding - known_emb).item()
            if dist < FACE_MATCH_THRESHOLD and dist < best_dist:
                best_match = known_path
                best_dist = dist

        if best_match:
            results[str(image_path.relative_to(image_folder))] = [(best_match, best_dist)]

    except Exception as e:
        print(f"Error processing {image_path}: {e}")

print(f"Face matches found for {len(results)} images.")
