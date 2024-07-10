from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from matplotlib import pyplot
import time

# load dataset
image_data = load_dataset(
    "keremberke/painting-style-classification", "full", split="train"
)

# define clip model for multimodal/contrastive image learning...and embeddings
embed_model = SentenceTransformer("clip-ViT-B-32")


# define helper functions
def map_filenames(sample):
    sample["image_file_path"] = sample["image_file_path"].split("/")[-1]
    return sample["image_file_path"]


def get_similar_images(query: str, dataset, k_images):
    stime = time.time()
    prompt = embed_model.encode(query)
    similarity_score, images_embeddings = dataset.get_nearest_examples(
        "embeddings", prompt, k=k_images
    )
    latency = time.time() - stime
    print(f"Retrieved {k_images} and similarity scores in {latency}")
    return similarity_score, images_embeddings


def image_grid(image_list):
    pyplot.figure(figsize=(20, 20))
    columns = 2
    for k in range(len(image_list)):
        image = image_list["image"][0]
        pyplot.subplot(len(image_list) / columns + 1, columns, k + 1)
        pyplot.imshow(image)


image_data = image_data.map(map_filenames)

image_data_embed = image_data.map(
    lambda example: {"embeddings": embed_model.encode(
        example["image"], device="cuda")},
    batched=True,
    batch_size=64,
)

# print features and display sample images
# print(image_data.features['labels'])
# image_data[0]['image']

image_data_embed.add_faiss_index(column="embeddings")

# text prompt or search term
prompt = embed_model.encode("men sitting together")

simscore, ret_images = image_data_embed.get_nearest_examples(
    "embeddings", prompt, k=5
)  # get similar images and scores

# ret_images[0]['image']
# print(score[0])

scores, similar_images = get_similar_images(
    "blue flowing river", image_data_embed, k_images=10
)

image_grid(similar_images)
