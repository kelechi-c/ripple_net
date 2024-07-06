import os
import rich
import argparse
from sentence_transformers import SentenceTransformer  # for clip embedding model
from datasets import load_dataset

# Intended functions
# -get all images, create index
# -search for/retrieve similar images
# -auto-labelling


def image_tagger(file_list: list):
    pass


class image_indexer:
    def __init__(self, image_folder):
        self.image_folder = image_folder
        # load dataset
        self.image_data = load_dataset("imagefolder", data_dir=image_folder)
        print(f"image dataset created from {self.image_folder}")
        # define clip model for multimodal/contrastive image learning...and embeddings
        print("Initializing clip model")
        self.embed_model = SentenceTransformer("clip-ViT-B-32")
        print(f"clip/embedding model initalized")

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


class ImageSearch:
    def __init__(self):
        pass
