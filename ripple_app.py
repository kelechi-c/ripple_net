import ripple
import streamlit as stl
from tqdm.auto import tqdm

# streamlit app
stl.set_page_config(
    page_title="Ripple",
)

stl.title("ripple search")
stl.write(
    "An app that uses text input to search for described images, using embeddings of selected image datasets. Uses contrastive learning models(CLIP) and the sentence transformers library"
)
stl.link_button(
    label="link to github and full library code",
    url="https://github.com/kelechi-c/ripple_net",
)

dataset = stl.selectbox(
    "choose huggingface dataset(bgger datasets take more time to embed..)",
    options=[
        "huggan/wikiart(1k)",
        "huggan/wikiart(11k)",
        "zh-plus/tiny-imagenet(110k)",
        "lambdalabs/naruto-blip-captions(1k)",
        "detection-datasets/fashionpedia(45k)",
    ],
)
# initalized global variables

embedded_data = None
embedder = None
text_search = None

ret_images = []
scores = []


if dataset and stl.button("embed image dataset"):
    with stl.spinner("Initializing and creating image embeddings from dataset"):
        embedder = ripple.ImageEmbedder(
            dataset, retrieval_type="text-image", dataset_type="huggingface"
        )

        embedded_data = embedder.create_embeddings(device="cpu")
        stl.success("Sucessfully embedded and dcreated image index")

if embedded_data is not None:
    text_search = ripple.TextSearch(embedded_data, embedder.embed_model)
    stl.success("Initialized text search class")

search_term = stl.text_input("Text description/search for image")

if search_term:
    with stl.spinner("retrieving images with description.."):
        scores, ret_images = text_search.get_similar_images(
            search_term, k_images=4)
        stl.success(f"sucessfully retrieved {len(ret_images)}")

for count, score, image in tqdm(zip(range(len(ret_images)), scores, ret_images)):
    stl.image(image["image"][count])
    stl.write(score)
