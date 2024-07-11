from sentence_transformers import SentenceTransformer, util
from PIL import Image as pillow
import os
from .utils import image_loader, get_all_images


class ImageTagger:
    def __init__(self, folder):
        self.clip_model = SentenceTransformer("clip-ViT-B-32")
        self.file_list = get_all_images(folder)
        # [os.path.join(os.path.dirname(file), file) for file in os.listdir(folder)]

    def auto_tagger(self, captions: list):
        for image in self.file_list:
            self.rename_image(image, captions)

    def rename_image(self, image_path, captions: list):
        # Encode image
        img_emb = self.clip_model.encode(image_loader(image_path))

        # Encode captions
        caption_emb = self.clip_model.encode(captions)

        # Find most similar caption
        similarities = util.cos_sim(img_emb, caption_emb)
        best_match = captions[similarities.argmax()]

        # Rename the file
        new_name = f"{best_match}.{image_path.split('.')[-1]}"
        os.rename(image_path, os.path.join(os.path.dirname(image_path), new_name))
        print(f"Renamed {image_path} to {new_name}")


# Define your captions and use this function on your image files
