import torch
import clip
from pathlib import Path
from PIL import Image, ExifTags
import pickle
from facenet_pytorch import MTCNN, InceptionResnetV1

# HEIC support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    print("pillow-heif not available — HEIC images may not load.")

# Fix image orientation
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

device = "cpu"
base_path = Path(__file__).resolve().parent
image_folder = base_path / "Images_Test"
known_faces_path = base_path / "known_faces"
valid_exts = [".jpg", ".jpeg", ".png", ".heic", ".heif"]

# Load models
clip_model, preprocess_clip = clip.load("ViT-B/32", device=device)
mtcnn = MTCNN(image_size=160, margin=0, device=device)
facenet_model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# FaceNet database
face_db = []
for path in known_faces_path.rglob("*"):
    if path.suffix.lower() not in valid_exts or path.name.startswith("._"):
        continue
    try:
        img = Image.open(path).convert("RGB")
        img = fix_image_orientation(img)
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
        img = fix_image_orientation(img)
        img_tensor = preprocess_clip(img).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = clip_model.encode_image(img_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        image_embeddings[str(path.relative_to(image_folder))] = embedding.cpu()
    except Exception as e:
        print(f"CLIP error: {path}: {e}")

with open("clip_embeddings.pkl", "wb") as f:
    pickle.dump(image_embeddings, f)

print("✅ Saved FaceNet and CLIP embeddings.")
