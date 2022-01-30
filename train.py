import tensorflow as tf

import numpy as np

from dataset_handler import (
    parse_coco_json,
    generate_dataset_masks,
    generate_dataset,
    preprocess_datapoint,
    Augment
)

from model_handler import (
    predict,
    generate_callbacks,
    UNet
)

from setup_handler import setup

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == "__main__":

    # Define the testing session variables
    EPOCHS = 30
    INTERVAL = 5
    BATCH_SIZE = 64
    BUFFER_SIZE = 1000
    MODEL_SIZE = 224
    EVALUATION_AMOUNT = 15
    DATASET = "wildfire"
    DATASET_SIZE = {
        "wildfire": (480, 640),
        "cells": (416, 416)
    }
    LABELS = {
        "wildfire": ["Background", "Smoke"],
        "cells": ["Background", "Platelets", "RBC", "WBC"]
    }

    # Setup the environment by removing all previously
    # generated predictions and checkpoints
    setup(
        dataset = DATASET, 
        url = "", 
        rm_images = True, 
        rm_checkpoints = True,
        rm_data = False
    )

    # Parse the training dataset into a CSV file
    parse_coco_json(
        input_path = f"./{DATASET}/train/_annotations.coco.json", 
        output_path = f"./annotations_training.csv", 
        image_path = f"./{DATASET}/train", 
        labels = LABELS[DATASET]
    )

    # Create image masks from the generated CSV dataset file
    # and save each one as a seperate file for later usage
    generate_dataset_masks(
        input_file = "./annotations_training.csv", 
        model_size = MODEL_SIZE
    )

    # Generate the dataset, and then define the
    # dataset batches as passing that dataset
    # through a data pipeline after preprocessing
    dataset = generate_dataset("./annotations_training.csv", MODEL_SIZE).map(
        preprocess_datapoint, 
        num_parallel_calls = tf.data.AUTOTUNE
    )

    dataset_batches = (
        dataset
        .cache()
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .map(Augment(seed = 100, rotation = 0.150))
        .prefetch(buffer_size = tf.data.AUTOTUNE)
    )

    # Initialize the UNet model and compile it
    model = UNet(
        input_shape = [MODEL_SIZE, MODEL_SIZE, 3],
        output_channels = 2
    )
    model.compile(optimizer = 'adam')

    # Create evaluation data to test model 
    # performance during training
    evaluation_data = dataset_batches.take(1)
    evaluation_data = (evaluation_data.unbatch()).take(EVALUATION_AMOUNT)

    # Create sample prediction(s) with the model
    predict(
        data = evaluation_data, 
        model = model, 
        path = "./predictions/example"
    )

    # Fit the model to the training data 'dataset_batches', 
    # saving the model weights and making predictions 
    # every 'INTERVAL' epochs
    model.fit(
        dataset_batches, 
        epochs = EPOCHS,
        verbose = 0,
        callbacks = generate_callbacks(
            interval = INTERVAL, 
            data = evaluation_data
        )
    )