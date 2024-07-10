import time
from .utils import image_grid
from datasets import Dataset
from sentence_transformers import SentenceTransformer


class TextSearch:
    def __init__(self, dataset: Dataset, model: SentenceTransformer):
        self.embed_model = model
        self.image_dataset = dataset
        self.k_images = None

    def get_similar_images(self, query: str, k_images=5):
        stime = time.time()
        self.k_images = k_images

        prompt = self.embed_model.encode(query)
        similarity_score, image_embeddings = self.image_dataset.get_nearest_examples(
            "embeddings", prompt, k=k_images
        )
        latency = time.time() - stime
        print("---")
        print(
            f"Retrieved {len(image_embeddings['image'])} and similarity scores in {latency:.4f}"
        )
        return similarity_score, image_embeddings

    def show_grid(self, images):
        image_grid(images)
