### ripple 
*(still in development)*
#### text/Image based search and retrieval for image datasets/files

#### Usage guide
- Installation

`pip install ripple_net` 

- Using ripple for text-based search

```python
from ripple import ImageEmbedder, TextSearch # import classes

embedder = ImageEmbedder('huggan/wikiart', retrieval_type='text-image', dataset_type='huggingface') # load from a huggingface image dataset or load from a local image directory

embedded_images = embedder.create_embeddings(device="cuda", batch_size=32) # could also use 'cpu' if CUDA-enabled GPU isn't available

text_search = TextSearch(embedded_data, embedder.embed_model)

scores, images = text_search.get_similar_images(query='painting of a river', k_images=10) # specify text query for image, and number of results to return

text_search.image_grid(images, scores) #dislay grid of returned images
```

- For image-based retrieval
.....
