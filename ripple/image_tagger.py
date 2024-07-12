from sentence_transformers import SentenceTransformer, util
import os
import shutil
from .utils import image_loader, get_all_images, latency
from tqdm.auto import tqdm


class ImageTagger:
    def __init__(self, folder, model_name="clip-ViT-B-32"):
        self.clip_model = SentenceTransformer(model_name)
        self.file_list = get_all_images(folder)
        print("Init ripple tagger")

    def auto_tagger(self, captions):
        for cap in captions:
            os.makedirs(cap, exist_ok=True)

        caption_emb = self.clip_model.encode(captions)

        for k, image in enumerate(tqdm(self.file_list)):
            self.rename_image(image, captions, caption_emb, k)

    @latency
    def rename_image(self, image_path, captions, caption_emb, k):
        try:
            img_emb = self.clip_model.encode(image_loader(image_path))
            similarities = util.cos_sim(img_emb, caption_emb)
            tag = captions[similarities.argmax()]

            file_ext = os.path.splitext(image_path)[1]
            new_name = f"{tag}_{k}{file_ext}"
            new_path = os.path.join(tag, new_name)

            shutil.move(image_path, new_path)
            print(f"Moved {image_path} to {new_path}")
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")


# Define your captions and use this function on your image files
