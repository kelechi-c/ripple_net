### ripple 
**(still in development)**
#### Search image datasets with words, or even label images automatically

Install with `pip install ripple_net` 
(I name things weirdly :/)

#### Usage guide

```python
from ripple import ImageEmbedder, ImageSearch # import classes

embedder = ImageEmbedder('huggan/wikiart', is_hfdataset=True) # load from a huggingface image dataset or load from a local image directory

embedded_images = embedder.create_embeddings(device="cuda", batch_size=32) # could also use 'cpu' if CUDA-enabled GPU isn't available

text_search =  ImageSearch(embedded_data, data_indexer.embed_model)
scores, images =text_search.get_similar_images(query='painting of a river', k_images=10) # specify text query for image, and number of results to return
text_search.image_grid(images, scores) # dislay grid of returned images
