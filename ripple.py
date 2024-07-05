from sentence_transformers import SentenceTransformer
from datasets import load_dataset

image_data = load_dataset(
    "keremberke/painting-style-classification", "full", split="train"
)


embed_model = SentenceTransformer("clip-ViT-B-32")


# define
def map_filenames(sample):
    sample["image_file_path"] = sample["image_file_path"].split("/")[-1]
    return sample["image_file_path"]


def map_embeddings(sample):
    example["embeddings"] = embed_model.encode(example["image"], device="cuda")
    return example["embeddings"]


image_data = image_data.map(map_filenames)

image_data_embed = image_data.map(
    lambda example: {"embeddings": embed_model.encode(
        example["image"], device="cuda")},
    batched=True,
    batch_size=64,
    num_proc=4,
)

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
