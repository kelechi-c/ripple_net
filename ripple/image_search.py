from typing import Literal
from datasets import Dataset
from PIL import Image as pillow
from .utils import image_grid, image_loader
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
import time


class ImageSearch:
    def __init__(
        self, embedded_dataset: Dataset, device: Literal["cuda", "cpu"]
    ) -> None:
        # initalize class and CLIP models
        self.model_id = "openai/clip-vit-large-patch14"
        self.device_id = device
        self.clip_model = AutoModelForZeroShotImageClassification.from_pretrained(
            self.model_id, device_map=self.device_id
        )
        self.clip_processor = AutoProcessor.from_pretrained(self.model_id)
        assert (
            "embeddings" in embedded_dataset.column_names
        ), "embeddings column missing in the input dataset. Ensure the dataset was embedded/indexed"
        self.embedded_data = embedded_dataset

    def image_search(self, input_img, k_count: int):
        if not isinstance(input_img, pillow):  # check if image type is PIL
            print("Image not in PIL format, converting..")
            input_img = image_loader(input_img)  # loads image in PIL format

        stime = time.time()
        pixel_values = self.clip_processor(images=input_img, return_tensors="pt")[
            "pixel_values"
        ]
        pixel_values = pixel_values.to(self.device_id)  # move tensors to device
        img_embed = self.clip_model.get_image_features(pixel_values)[0]
        img_embed = img_embed.detach().cpu().numpy()

        scores, retrieved_images = self.embedded_data.get_nearest_examples(
            "embeddings", img_embed, k=k_count
        )
        exec_time = stime - time.time()
        print(f"Retrieved {len(retrieved_images)} in {exec_time} seconds")
        return scores, retrieved_images

    def show_grid(self, images):
        image_grid(images)
