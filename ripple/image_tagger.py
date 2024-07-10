from sentence_transformers import SentenceTransformer, util
from PIL import Image as pillow
from ripple import image_loader
import os

# Load model
clip_model = SentenceTransformer("clip-ViT-B-32")


def rename_image(image_path, captions):
    # Encode image
    img_emb = clip_model.encode(image_loader(image_path))

    # Encode captions
    caption_emb = clip_model.encode(captions)

    # Find most similar caption
    similarities = util.cos_sim(img_emb, caption_emb)
    best_match = captions[similarities.argmax()]

    # Rename the file
    new_name = f"{best_match}.{image_path.split('.')[-1]}"
    os.rename(image_path, os.path.join(os.path.dirname(image_path), new_name))


# Define your captions and use this function on your image files
