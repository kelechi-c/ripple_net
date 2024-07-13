## ripple_net *(wip)*

A library for text/image based search/retrieval for image datasets and files. Uses multimodal AI techniques/models like vector embeddings and CLIP.

## Install

```bash
$ pip install ripple_net
```

## Usage

- For text description-based search

```python
from ripple import ImageEmbedder, TextSearch # import classes

# load from a huggingface image dataset or load from a local image directory
embedder = ImageEmbedder('huggan/wikiart', retrieval_type='text-image', dataset_type='huggingface') 

# could also use 'cpu' if CUDA-enabled GPU isn't available
embedded_images = embedder.create_embeddings(device="cuda", batch_size=32)

# initialize text - image search class
text_search = TextSearch(embedded_images, embedder.embed_model)

# specify text/search query for image, and number of results to return
scores, images = text_search.get_similar_images(query='painting of a river', k_images=10) 

images['image'][0].show()
```

- For image-based retrieval(image-image search)

```python
from ripple import ImageEmbedder, ImageSearch, image_loader

 # load dataset and initialize embedding class
embedder = ImageEmbedder('lambdalabs/naruto-blip-captions', retrieval_type='image-image', dataset_type='huggingface',  device='cuda',
)

# generate embeddings
embedded_images = embedder.create_embeddings(device="cuda", batch_size=32)

# init image search class
image_search = ImageSearch(embedded_images, embedder.embed_model)

# retrieve similar images with image input
input_image = image_loader('katara.png') # use library function to load image in PIL format

scores, images = image_search.image_search(input_img=input_image, k_images=5) # specify input image, and number of results to return

# dislay one of retrieved images
images['image'][0].show()
# or using notebooks => images['image'][0]
```

- For auto image tagging/renaming

```python
from ripple import ImageTagger

# initialize the class with folder of choice
folder = '/kaggle/working/images/drawings'

tagger = ImageTagger(folder)

# captions to label with
captions = ['humans', 'animals', 'plants','land']

tagger.auto_tagger(captions) # rename all images and move to folders
```

## Todo

- [ ] direct CLI usage

## Acknowledgement

- <a href="https://sbert.net/">Sentence transformers </a> library by UKPLabs and Huggingface transformers.
- <a href="https://huggingface.co/blog/not-lain/image-retriever">Image search engine</a>: article by <a href="https://github.com/not-lain">not-lain </a>
- <a href="https://openai.com/index/clip/">CLIP (Contrastive Language–Image Pre-training)</a> research by OpenAI.
