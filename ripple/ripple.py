import os
import rich
import time
from sentence_transformers import SentenceTransformer  # for clip embedding model
from datasets import Dataset, load_dataset
from functools import wraps
from matplotlib import pyplot
from tqdm.auto import tqdm

# Intended functions
# -get all images, create index > done
# -search for/retrieve similar images > done
# -auto-labelling


def latency_check(unit="seconds"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()

            elapsed_time = end_time - start_time

            if unit == "milliseconds":
                elapsed_time *= 1000
                time_unit = "ms"
            else:
                time_unit = "s"

            print(
                f"latency for {func.__name__} => {elapsed_time:.4f} {time_unit}")
            return result

        return wrapper

    return decorator


def image_tagger(file_list: list):
    pass


class image_indexer:
    @latency_check
    def __init__(self, image_folder):
        self.image_folder = image_folder
        # load dataset
        self.image_data = load_dataset("imagefolder", data_dir=image_folder)
        print(f"image dataset created from {self.image_folder}")

        # define clip model for multimodal/contrastive image learning...and embeddings
        print("Initializing clip model")
        self.embed_model = SentenceTransformer("clip-ViT-B-32")
        print(f"clip/embedding model initalized")

    def map_filenames(sample):
        sample["image_file_path"] = sample["image_file_path"].split("/")[-1]
        return sample["image_file_path"]

    @latency_check
    def create_embeddings(self):
        # create embedding class
        image_data_embed = image_data.map(
            lambda example: {
                "embeddings": embed_model.encode(example["image"], device="cuda")
            },
            batched=True,
            batch_size=64,
            num_proc=4,
        )
        image_data_embed.add_faiss_index(column="embeddings")
        return image_data_embed

    def get_similar_images(query: str, dataset, k_image):
        stime = time.time()
        prompt = model.encode(query)
        similarity_score, images_embeddings = dataset.get_nearest_examples(
            "embeddings", prompt, k=k_images
        )
        latency = time.time() - stime
        print(f"Retrieved {k_image} and similarity scores in {latency}")
        return similarity_score, images_embeddings

    def image_grid(image_list):
        pyplot.figure(figsize=(20, 20))
        columns = 2
        for k in range(len(image_list)):
            image = image_list["image"][0]
            pyplot.subplot(len(image_list) / columns + 1, columns, k + 1)
            pyplot.imshow(image)


class ImageSearch:
    def __init__(self, dataset: Dataset, model: SentenceTransformer):
        self.embed_model = model
        self.image_dataset = dataset

    @latency_check
    def get_similar_images(self, query: str, k_images=10):
        stime = time.time()
        prompt = model.encode(query)
        similarity_score, images_embeddings = dataset.get_nearest_examples(
            "embeddings", prompt, k=k_images
        )
        latency = time.time() - stime
        print(f"Retrieved {k_image} and similarity scores in {latency}")
        return similarity_score, images_embeddings

    def image_grid(self, image_list, scores):
        pyplot.figure(figsize=(20, 20))
        columns = 2
        for k, score in tqdm(zip(range(len(image_list), scores)), color="blue"):
            image = image_list["image"][k]
            pyplot.subplot(len(image_list) / columns + 1, columns, k + 1)
            pyplot.title(f"")
            pyplot.imshow(image)
