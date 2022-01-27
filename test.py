import tensorflow as tf

import numpy as np

from dataset_handler import (
    parse_coco_json,
    generate_dataset_masks,
    generate_dataset,
    preprocess_datapoint
)

from model_handler import (
    batch_predict,
    UNet
)

from setup_handler import setup

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == "__main__":

    # Define the training session variables
    BATCH_SIZE = 64
    BUFFER_SIZE = 1000
    MODEL_SIZE = 224
    CKPT_INDEX = 30
    DATASET = "wildfire"
    SPLIT = "test"
    DATASET_SIZE = {
        "wildfire": (480, 640),
        "cells": (416, 416)
    }
    LABELS = {
        "wildfire": ["Background", "Smoke"],
        "cells": ["Background", "Platelets", "RBC", "WBC"]
    }

    # Setup the environment by removing all previously
    # generated predictions
    setup(
        dataset = DATASET, 
        url = "", 
        rm_images = False, 
        rm_checkpoints = False,
        rm_data = False
    )

    # Initialize the UNet model and compile it
    model = UNet(
        input_shape = [MODEL_SIZE, MODEL_SIZE, 3],
        output_channels = 2
    )
    model.load_weights(f"./checkpoints/checkpoint-{CKPT_INDEX:03d}.ckpt").expect_partial()

    # Parse the testing dataset into a CSV file
    parse_coco_json(
        input_path = f"./{DATASET}/{SPLIT}/_annotations.coco.json", 
        output_path = f"./annotations_testing.csv", 
        image_path = f"./{DATASET}/{SPLIT}", 
        labels = LABELS[DATASET]
    )

    # Create image masks from the generated CSV dataset file
    # and save each one as a seperate file for later usage
    generate_dataset_masks(
        input_file = "./annotations_testing.csv", 
        image_shape = DATASET_SIZE[DATASET],
        model_size = MODEL_SIZE
    )

    # Generate the dataset, and then define the
    # dataset batches as passing that dataset
    # through a data pipeline after preprocessing
    dataset = generate_dataset("./annotations_testing.csv", MODEL_SIZE).map(
        preprocess_datapoint, 
        num_parallel_calls = tf.data.AUTOTUNE
    )

    dataset_batches = (
        dataset
        .cache()
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .prefetch(buffer_size = tf.data.AUTOTUNE)
        .shuffle(len(dataset))
    )

    # Test the model with the dataset batches
    batch_predict(
        data = dataset_batches, 
        model = model,
        path = "./predictions/testing"
    )