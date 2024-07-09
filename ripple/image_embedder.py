import os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from utils import latency
from typing import Literal


class ImageEmbedder:
    @latency
    def __init__(
        self,
        image_data: str,
        retrieval_type: Literal["text-image", "image-image"],
        is_hfdataset=False,
    ):
        assert retrieval_type in [
            "text_image",
            "image_image",
        ], "retrieval/search type must be either 'image-image' or 'text-image'"

        # initial variables
        self.image_dataset = None
        self.data_path = image_data
        self.is_hfdataset = is_hfdataset
        self.retrieval_type = retrieval_type
        self.embed_model = None
        self.processor_model = None
        self.device = None

        # load dataset
        if not self.is_hfdataset:
            self.image_dataset = (
                load_dataset("imagefolder", data_dir=image_data, split="train")
                if os.path.isdir(image_data)
                else load_dataset(
                    "imagefolder", data_dir=".", split="train"
                )  # load from local root folder
            )
        else:
            self.image_dataset = load_dataset(
                image_data, split="train"
            )  # load from huggingface dataset instead

        print(f"image dataset created from {image_data}")
        print("----")
        # define clip model for multimodal/contrastive image learning...and embeddings
        print("Initializing CLIP model")
        print("....")

        # load model based on retrieval type
        if self.retrieval_type == "text-image":
            self.embed_model = SentenceTransformer("clip-ViT-B-32")

        elif self.retrieval_type == "image-image":
            self.processor_model = AutoProcessor.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
            self.embed_model = AutoModelForZeroShotImageClassification(
                "openai/clip-vit-large-patch14"
            )

        print(f"clip/embedding model -[{self.embed_model}] initialized")

    @latency
    def create_embeddings(self, device: Literal["cuda", "cpu"], batch_size: int = 32):
        assert device in [
            "cuda", "cpu"], "Wrong device id, must be 'cuda' or 'cpu'"

        image_embeddings = None
        self.device = device

        # map embedding function to the dataset
        if self.retrieval_type == "text-image":
            image_embeddings = self.image_dataset.map(
                lambda example: {
                    "embeddings": self.embed_model.encode(
                        example["image"], device=device
                    )
                },
                batched=True,
                batch_size=batch_size,
            )

        elif self.retrieval_type == "image-image":
            image_embeddings = self.image_dataset.map(self._embed_image_batch)

        image_embeddings.add_faiss_index(column="embeddings")
        print(
            f"vector embeddings and faiss-index created for {self.data_path}")
        return image_embeddings

    def _embed_image_batch(self, batch):
        pixels = self.processor_model(images=batch["image"], return_tensors="pt")[
            "pixel_values"
        ]
        pixels = pixels.to(self.device)

        image_embedding = self.embed_model.get_image_features(pixels)
        batch["embeddings"] = image_embedding
        return batch
