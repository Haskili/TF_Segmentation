import tensorflow as tf

import numpy as np
import pandas as pd

import PIL
from PIL import Image, ImageDraw, ImageFont

import json
import os


def parse_coco_json(input_path, output_path, image_path, labels):
    """
    Read a COCO JSON formatted annotations file corrosponding to a set
    of images, and create a CSV file of that data in an alternate format, 
    (e.g. image_path, label, xmin, ymin, xmax, ymax).

    Arguments:
            input_path (str): Path to the JSON file
            output_path (str): Path to a output file that will be created
            image_path (str): Path to a directory of images
            labels (list[str]): The possible labels within the dataset

    Returns:
            N/A

    Raises:
            N/A
    """

    # Iterate through each row of data in the input file
    with open(input_path, "r") as input_file:
        json_data = json.load(input_file)

        # Seperate out the import information
        catagory_list   = json_data["categories"]
        image_list      = json_data["images"]
        annotation_list = json_data["annotations"]

        # Create a dictionary of valid examples (e.g. images exist)
        image_dict = {}
        for image in image_list:
            if os.path.exists(f"{image_path.rstrip('/')}/{image['file_name']}"):
                image_dict[image["id"]] = image["file_name"]

        # Using the valid images dictionary, create a dictionary
        # of examples wherein the values are the valid examples
        # for that specific label
        annotation_dict = {}
        for annotation in annotation_list:
            if annotation["image_id"] in image_dict:
                annotation_dict[annotation["id"]] = {
                    "image" : annotation["image_id"],
                    "label" : annotation["category_id"],
                    "bounds" : annotation["bbox"]
                }

        # Begin writing the output file
        with open(output_path, 'w') as output_file:

            # Write file header
            output_file.write("image_path,label,xmin,ymin,xmax,ymax\n")

            # Write each annotation given into the output file
            for annotation_id, annotation in annotation_dict.items():
                if labels[annotation['label']] != "Background":
                    labels[annotation['label']] = "Object"

                output_data = [
                    f"{image_path}/{image_dict[annotation['image']]}",
                    f"{labels[annotation['label']]}",
                    f"{int(annotation['bounds'][0])}",
                    f"{int(annotation['bounds'][1]) + int(annotation['bounds'][3])}",
                    f"{int(annotation['bounds'][0]) + int(annotation['bounds'][2])}",
                    f"{int(annotation['bounds'][1])}",
                ]
                output_file.write(f"{','.join([str(data) for data in output_data])}\n")


def generate_mask(boxes: list, original_size: tuple, model_size: int):
    """
    Generate and return a mask image given a list containing a
    bounding box for every object within the image.

    Arguments:
            boxes (list): List containing each object's bounding box
            original_size (tuple): Original size of image being masked
            model_size (int): Input size of images for the model

    Returns:
            mask (numpy) (shape = (model_size, model_size, 1))

    Raises:
            N/A
    """

    # Initialize an empty mask image
    mask = Image.fromarray(np.zeros(original_size))
    drawing = ImageDraw.Draw(mask)    

    # Draw each bounding box onto the mask image
    for box in boxes:
        x_min, y_min, x_max, y_max = box

        # Create an empty mask image and draw the specified polygon
        drawing.polygon(
            [
                (x_min, y_min), 
                (x_min, y_max), 
                (x_max, y_max), 
                (x_max, y_min), 
                (x_min, y_min)
            ],
            fill  = 'white',
            outline = None
        )

    # Resize the mask image and format it for return to caller
    mask = np.array(mask.resize((model_size, model_size))) // 255.0
    mask[mask < 0.0] = 1.0
    return mask.reshape((model_size, model_size, 1))


def generate_dataset_masks(input_file: str, image_shape: tuple, model_size: int):
    """
    Generate and save a mask image for every single image inside
    the given dataset file 'input_file'.

    Arguments:
            input_file (str): Path to the CSV dataset file
            image_shape (tuple): Original size of image being masked
            model_size (int): Input size of images of the model

    Returns:
            N/A

    Raises:
            N/A
    """

    # Initialize an empty dictionary
    images = {}

    # Collect each datapoint inside the CSV and store them
    # according to their associated image example (path)
    for index, row in pd.read_csv(input_file).iterrows():
        (image_path, label, x_min, y_min, x_max, y_max) = row

        # If the image isn't in the dictionary, 
        # initialize it's key value as an empty list
        if image_path not in images:
            images[image_path] = []

        # Append the list of boxes for the image
        images[image_path].append([x_min, y_min, x_max, y_max])

    # For each image, generate the a mask image including each
    # object within the image and save the final result
    for image_path in images.keys():
        mask = generate_mask(images[image_path], image_shape, model_size)
        np.save(f"./masks/{image_path.split('/')[-1]}", mask)


def generate_dataset(input_path: str, model_size: int):
    """
    Read in a CSV dataset file, 'input_path', and return a zipped dataset
    of the image, label, and mask for each example therein.

    Arguments:
            input_path (str): Path to the CSV file of example data
            model_size (int): Input size of images of the model

    Returns:
            A zipped tf.data.Dataset of the images, labels, and masks

    Raises:
            N/A
    """

    # Create empty structures to hold
    # the individual dataset components
    images = []
    labels = []
    masks  = []
    seen = {}

    # Iteratively append the lists as required
    for index, row in pd.read_csv(input_path).iterrows():
        (image_path, label, xmin, ymin, xmax, ymax) = row
        if image_path in seen:
            continue

        image = tf.convert_to_tensor(np.array(Image.open(image_path).resize((model_size, model_size))))
        mask = tf.convert_to_tensor(np.load(f"./masks/{image_path.split('/')[-1]}.npy")) 

        images.append(image)
        labels.append(label)
        masks.append(mask)
        seen[image_path] = 1

    # Convert each list to a tf.data.Dataset object
    images = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(np.array(images)))
    labels = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(np.array(labels)))
    masks = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(np.array(masks)))

    # Return the zip of all three tf.data.Dataset objects
    return tf.data.Dataset.zip((images, labels, masks))


def normalize(input_image, input_mask, sub_operation = False):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    if sub_operation:
        input_mask -= 1.0

    return input_image, input_mask


def preprocess_datapoint(image, label, mask):
    input_image, input_mask = normalize(image, mask)
    return input_image, input_mask


class Augment(tf.keras.layers.Layer):
    def __init__(self, seed = 50, rotation = 0.035, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rotation = rotation

        self.augment_inputs = tf.keras.Sequential([
            tf.keras.layers.RandomFlip(mode = "horizontal", seed = seed),
            tf.keras.layers.RandomRotation(self.rotation, seed = seed)
        ])

        self.augment_masks = tf.keras.Sequential([
            tf.keras.layers.RandomFlip(mode = "horizontal", seed = seed),
            tf.keras.layers.RandomRotation(self.rotation, seed = seed)
        ])


    def call(self, inputs, masks):
        inputs = self.augment_inputs(inputs)
        masks = self.augment_masks(masks)
        return inputs, masks