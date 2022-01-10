import os
import requests

from zipfile import ZipFile
from glob import glob


def setup(dataset = "", url = "", rm_images = False, rm_checkpoints = False, rm_data = False):

    # Ensure that the required directories are present
    for directory in ["./predictions", "./masks", "./checkpoints", f"./{dataset}"]:
        if not os.path.exists(f"./predictions"):
               os.mkdir(f"./predictions")

    # Delete all training, testing, example, and mask images
    if rm_images:
        training_images = glob("./predictions/training_*_*.png")
        testing_images = glob("./predictions/testing_*.png")
        example_images = glob("./predictions/example_*.png")
        mask_images = glob("./masks/*.npy")
        all_images = training_images + testing_images + example_images + mask_images

        for image in all_images:
            try:
                os.remove(image)

            except Exception as exception:
                print(exception)

    # Delete all checkpoints
    if rm_checkpoints:
        for file in glob("./checkpoints/*"):
            try:
                os.remove(file)

            except Exception as exception:
                print(exception)

    # Delete and re-download the dataset
    if rm_data:
        if url == "":
            print("Empty URL, cannot re-download dataset, exiting...")
            os.exit(1)

        # Remove all sub-directories inside the dataset directory
        for directory in glob(f"./{dataset}/*"):
            try:
                os.rmdir(directory)

            except Exception as exception:
                print(exception)

        # Download and extract the dataset
        request = requests.get(url)
        with open("./dataset.zip", "wb") as zip_file:
            zip_file.write(request.content)

        with zipfile.ZipFile("./dataset.zip", "r") as zip_reference:
            zip_reference.extractall(f"./{dataset}")

        # Remove the lingering zip file and any annotation files
        for file in ["./dataset.zip"] + glob("./annotations_*.csv"): 
            try:
                os.remove(file)

            except Exception as exception:
                print(exception)