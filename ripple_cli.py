import os
import glob
import argparse
from tqdm.auto import tqdm
import ripple

embedder = None


def main():
    # args
    parser = argparse.ArgumentParser(
        description="ripple: cli script for text-image, amd image similarity search :) "
    )
    parser.add_argument("-f", "--folder", help="the folder to load images from")
    parser.add_argument(
        "-a", "--all", action="store_true", help="use all image files on the device"
    )
    args = parser.parse_args()

    print(f"Loading images from folder{args.folder}")
    if args.all:  # loads all the images on the device
        print(f"getting all image files")
        file_list = ripple.get_all_images("/home/")
        embedder = ripple.ImageEmbedder(file_list, retrieval_type="text-image")

    else:
        embedder = ripple.ImageEmbedder(args.folder, retrieval_type="text-image")

    print(f"creating embeddings using {embedder.embed_model}...")
    embedded_data = embedder.create_embeddings(device="cpu")

    text_search = ripple.TextSearch(embedded_data, embedder.embed_model)
    scores, ret_images = text_search.get_similar_images(
        "girl wearing blue clothes", k_images=3
    )

    for score, image in tqdm(zip(scores, ret_images)):
        print(image.filename)
        print(score)
        print("----")

    text_search.show_grid(ret_images)
    display_image = ripple.image_loader(ret_images["image"][0])
    display_image.show()


if __name__ == "__main__":
    main()
