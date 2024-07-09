from math import pi
from operator import ipow
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from datasets import load_dataset
import numpy as np
from PIL import Image as pillow
from matplotlib import pyplot as plt
# from functools import isinstance

image_folder = ""
dataset_id = "huggan/few-shot-art-painting"
model_id = "openai/clip-vit-large-patch14"
batch_size = 32
devices = ["cuda", "cpu"]
device_id = "cuda"

sample_data = load_dataset(dataset_id, split="train")

# sample_data

clip_processor = AutoProcessor.from_pretrained(model_id)

clip_model = AutoModelForZeroShotImageClassification(model_id, device_map="cuda")


def image_loader(img):
    if isinstance(img, np.ndarray):
        return pillow.fromarray(img)

    elif isinstance(img, str):
        return pillow.open(img)

    elif isinstance(img, pillow):
        return img


def grid(images):
    f, ax = plt.subplots(2, 2)
    for index in range(len(images)):
        k, v = index // 2, index % 2
        # ax[k, v].set_title(images["image"][index].filename)
        ax[k, v].imshow(images["image"][index])
        ax[k, v].axis("off")
    plt.show()


def embed_image_batch(batch):
    pixels = clip_processor(images=batch["image"], return_tensors="pt")["pixel_values"]
    pixels = pixels.to(device_id)
    image_embedding = clip_model.get_image_features(pixels)
    batch["embeddings"] = image_embedding
    return batch


embedded_data = sample_data.map(embed_image_batch, batched=True, batch_size=batch_size)
embedded_data.add_faiss_index("embeddings")


def image_search(input_img, k_count: int):
    if not isinstance(input_img, pillow):  # check if image type is PIL
        input_img = image_loader(input_img)  # loads pil image

    pixel_values = clip_processor(images=input_img, return_tensors="pt")["pixel_values"]
    pixel_values = pixel_values.to(device_id)
    img_embed = clip_model.get_image_features(pixel_values)[0]
    img_embed = img_embed.detach().cpu().numpy()

    scores, retrieved_images = embedded_data.get_nearest_examples(
        "embeddings", img_embed, k=k_count
    )

    return retrieved_images


similar_images = image_search()
