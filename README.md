# Image Search App with CLIP and FaceNet

This project is a local Streamlit application for searching personal image collections using natural language prompts (via CLIP) and recognizing known faces (via FaceNet).

## Features

- CLIP-based semantic image search
- Face detection and face matching using MTCNN and FaceNet
- HEIC and HEIF image format support using pillow-heif
- Rotation correction for mobile images using EXIF data
- Streamlit interface for interactive search and image viewing
- Image download button for each result

## Project Structure

image-search-app/
├── app.py                  # Streamlit UI
├── save_embeddings.py      # Precomputes embeddings
├── clip_embeddings.pkl     # CLIP image embeddings
├── face_db.pkl             # FaceNet embeddings
├── Images_Test/            # Searchable images
├── known_faces/            # Reference faces
├── requirements.txt        # Dependencies
└── README.md               # Project documentation

## Setup Instructions

1. **Clone the repository**

   Open a terminal and run:

git clone https://github.com/yourusername/image-search-app.git
cd image-search-app


2. **Install dependencies**

It's recommended to use a virtual environment.

pip install -r requirements.txt


3. **Add your images**

- Put searchable images inside the `Images_Test/` folder.
- Place labeled reference face images in the `known_faces/` folder.

4. **Generate embeddings**

Run the script to create the required `.pkl` files:

python save_embeddings.py


This will generate `clip_embeddings.pkl` and `face_db.pkl`.

5. **Run the Streamlit app**

streamlit run Image_Search_App.py


## Example Prompts to Try

- people at dinner
- Award Ceremony
- Toronto Tech Conference
- beach picnic

## Notes

- Re-run `save_embeddings.py` only when you change or add images.
- Landscape images are rotated automatically for consistent display.
- HEIC/HEIF files are supported if `pillow-heif` is installed.
- Search results include similarity scores and allow image downloads.

## Dependencies

The app uses the following Python packages:

- streamlit
- torch
- clip (via GitHub: https://github.com/openai/CLIP)
- facenet-pytorch
- pillow
- pillow-heif
- numpy

To install them:
pip install -r requirements.txt

