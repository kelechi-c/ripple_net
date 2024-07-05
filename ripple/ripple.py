import os
import rich
import argparse
from sentence_transformers import SentenceTransformer  # for clip embedding model
from datasets import load_dataset
from

# Intended functions
# -get all images, create index
# -search for/retrieve similar images
# -auto-labelling


def image_tagger(file_list: list):
    pass


class image_indexer:
    def __init__(self, image_folder):
        self.image_folder = image_folder

    def create_embeddings(self):
        pass
