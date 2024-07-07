import os
import time
import math
from sentence_transformers import SentenceTransformer  # for clip embedding model
from datasets import Dataset, load_dataset
from functools import wraps
from matplotlib import pyplot as plt

# Intended functions
# -get all images, create index > done
# -search for/retrieve similar images > done
# -auto-labelling


def latency(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(
            f"latency => {func.__name__}: {end_time - start_time:.4f} seconds")
        return result

    return wrapper


class ImageEmbedder:
    @latency
    def __init__(self, image_data: str, is_hfdataset=False):
        self.image_dataset = image_data
        self.is_hfdataset = is_hfdataset
        # load dataset
        if not self.is_hfdataset:
            self.image_data = (
                load_dataset(
                    "imagefolder", data_dir=self.image_data, split="train")
                if os.path.isdir(self.image_data)
                else load_dataset(
                    "imagefolder", data_dir=".", split="train"
                )  # load from local folder
            )
        else:
            self.image_data = load_dataset(
                self.image_dataset, split="train"
            )  # load from huggingface dataset instead

        print(f"image dataset created from {self.image_dataset}")
        print("----")
        # define clip model for multimodal/contrastive image learning...and embeddings
        print("Initializing clip model")
        print("....")
        self.embed_model = SentenceTransformer("clip-ViT-B-32")
        print(f"clip/embedding model initalized")

    @latency
    def create_embeddings(self, device, batch_size):
        # create embedding class
        assert device in [
            "cuda", "cpu"], "Wrong device id, must be 'cuda' or 'cpu'"
        image_data_embed = self.image_data.map(
            lambda example: {
                "embeddings": self.embed_model.encode(example["image"], device=device)
            },
            batched=True,
            batch_size=batch_size,
        )
        image_data_embed.add_faiss_index(column="embeddings")
        return image_data_embed


class ImageSearch:
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

    def display_imagegrid(self, image_list):
        try:
            columns = 4
            rows = math.ceil(self.k_images / columns)

            fig, axs = plt.subplots(math.floor(
                rows), columns, figsize=(10, 10))

            # Flatten the 2D array of subplots into a 1D array
            axs = axs.flatten()

            for k, ax in enumerate(axs):
                ax.imshow(image_list["image"][k])
                ax.axis("off")

            plt.show()
            plt.axis("off")

        except Exception as e:
            print(e)
            print("loading grid....")
