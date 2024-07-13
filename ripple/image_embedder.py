from datasets import Dataset, load_dataset, Image
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from .utils import latency, get_all_images
from typing import Literal
import os


class ImageEmbedder:
    def __init__(
        self,
        image_data: str,
        retrieval_type: Literal["text-image", "image-image"],
        dataset_type: Literal["huggingface", "image folder"],
        device: Literal["cuda", "cpu"],
    ):
        assert retrieval_type in [
            "text-image",
            "image-image",
        ], "retrieval/search type must be either 'image-image' or 'text-image'"

        # initial variables
        # self.image_dataset = None
        self.dataset_type = dataset_type
        self.data_path = image_data
        self.retrieval_type = retrieval_type
        self.embed_model = None
        self.processor_model = None
        self.device = device

        # load dataset for different dataset types
        print(f"Loading huggingface dataset from {image_data}")
        if self.dataset_type == "huggingface":
            self.image_dataset = load_dataset(
                image_data, split="train"
            )  # load from huggingface dataset instead

        elif self.dataset_type == "image folder":
            if os.path.exists(self.data_path):
                image_list = get_all_images(image_data)
                self.image_dataset = Dataset.from_dict(
                    {"image": image_list}
                ).cast_column("image", Image())

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
            self.embed_model = AutoModelForZeroShotImageClassification.from_pretrained(
                "openai/clip-vit-large-patch14", device_map=self.device
            )

        print(f"clip/embedding model -[{self.embed_model}] initialized")

    @latency
    def create_embeddings(self, device: Literal["cuda", "cpu"], batch_size: int = 32):
        assert device in [
            "cuda",
            "cpu",
        ], "Wrong id, device must must be either 'cuda' or 'cpu'"

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
        print(f"Image vector embeddings and FAISS-index created for {self.data_path}")
        return image_embeddings

    def _embed_image_batch(self, batch):
        pixels = self.processor_model(images=batch["image"], return_tensors="pt")[
            "pixel_values"
        ]
        pixels = pixels.to(self.device)

        image_embedding = self.embed_model.get_image_features(pixels)
        batch["embeddings"] = image_embedding

        return batch
