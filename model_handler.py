import datetime
import sys

import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix

import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt


def draw_images(image, mask, prediction, segmentation, path):

    # Declare the figure
    plt.figure(figsize = (10, 10))

    # Declare the components of the figure
    fig, ((axs_A, axs_B), (axs_C, axs_D)) = plt.subplots(
        nrows = 2, 
        ncols = 2, 
        sharex = True, 
        sharey = True
    )

    # For each image, plot it on the figure
    axs_A.imshow(tf.keras.utils.array_to_img(image))
    axs_B.imshow(tf.keras.utils.array_to_img(mask))
    axs_C.imshow(tf.keras.utils.array_to_img(segmentation))
    axs_D.imshow(tf.keras.utils.array_to_img(prediction))

    # Hide the tickmark labels and setup the figure layout
    for axs in [axs_A, axs_B, axs_C, axs_D]:
        axs.xaxis.set_ticklabels([])
        axs.yaxis.set_ticklabels([])

    fig.tight_layout(pad = 0.0)

    # Save and reset the resulting figure
    plt.savefig(path)
    plt.close('all')


def apply_segmentation(image, prediction):
    indices = prediction[:, :, 0] != 0
    segmentation = np.zeros(image.shape)
    segmentation[indices] = image[indices]
    return segmentation


def predict(data, model, path):
    for index, (image, mask) in enumerate(data):

        # Predict on the given input data 'image',
        # and format the resulting output to get 
        # the most likely classification for each pixel
        prediction = model(np.array([image]))
        prediction = tf.argmax(prediction, axis = -1)
        prediction = prediction[..., tf.newaxis][0]

        # Create the segmentation image from the 
        # predicted mask and the original input
        segmentation = apply_segmentation(image, prediction)

        # Generate the result image, which includes the original input,
        # true segmentation mask, predicted segmentation mask, and the 
        # applied predicted segmentation onto the original input
        draw_images(image, mask, segmentation, prediction, f"{path}_{index:03d}.png")


def batch_predict(dataset, model, path, track_progress = True):

    # If required, setup a progress bar
    if track_progress:
        progress = tqdm(
            unit = "batches", 
            total = tf.data.experimental.cardinality(dataset).numpy(),
            ncols = 75
        )

    # Create a variable to track the amount of processed images
    # (NOTE: required because batching can have remainders)
    image_index = 0

    # Iterate through each batch of data in 'dataset'
    for image_batch, mask_batch in dataset:

        # Predict on the batch of input data,
        # and format the resulting outputs to
        # get the most likely classification for
        # each pixel in each prediction
        predictions = model(image_batch)
        predictions = tf.argmax(predictions, axis = -1)
        predictions = predictions[..., tf.newaxis]

        # Apply the predicted segmentations onto
        # their respective original input images
        # to get segmentation images
        segmentations = np.array([
            apply_segmentation(image, prediction) 
            for image, prediction in zip(image_batch, predictions)
        ])

        # Create the batch results iterator
        batch_results = zip(image_batch, mask_batch, predictions, segmentations)

        # Generate each result image, which includes the original input,
        # true segmentation mask, predicted segmentation mask, and the 
        # applied predicted segmentation onto the original input
        for image, mask, prediction, segmentation in batch_results:
            draw_images(image, mask, prediction, segmentation, f"{path}_{image_index:03d}.png")
            image_index += 1

        # If the dataset progress bar is active, 
        # update it for the completed batch 
        if track_progress:
            progress.update(1)



class CheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, directory, frequency, interval, optimize, monitor, mode, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.directory = directory
        self.frequency = frequency
        self.interval = interval
        self.optimize = optimize
        self.monitor = monitor
        self.mode = mode

        self.optimal_value = None


    def on_epoch_end(self, epoch, logs = None):
        self.save_checkpoint(
            frequency = "epoch", 
            index = (epoch + 1), 
            logs =  logs
        )


    def on_train_batch_end(self, batch, logs = None):
        self.save_checkpoint(
            frequency = "batch", 
            index = (batch + 1), 
            logs = logs
        )


    def save_checkpoint(self, frequency, index, logs):

        # If the saving frequency isn't set to 'frequency'
        if self.frequency != frequency:
            return

        # If the current 'index' doesn't match the saving interval
        if (index < self.interval) or (index % self.interval):
            return

        # If the callback is set to save only the "best" results
        if self.optimize:

            # If this is the first time checking, save it as the initial result
            if self.optimal_value == None:
                self.optimal_value = logs[self.monitor]
                (self.model).save_weights(f"{self.directory}/checkpoint_{index:03d}")

            # Else-if the qualification of "best" is maximization of the 
            # 'monitor' value and the current value meets that criteria
            elif self.mode == "max" and logs[self.monitor] > self.optimal_value:
                self.optimal_value = logs[self.monitor]
                (self.model).save_weights(f"{self.directory}/checkpoint_{index:03d}")

            # Else-if the qualification of "best" is minimization of the 
            # 'monitor' value and the current value meets that criteria
            elif self.mode == "min" and logs[self.monitor] < self.optimal_value:
                self.optimal_value = logs[self.monitor]
                (self.model).save_weights(f"{self.directory}/checkpoint_{index:03d}")

        # Else we save regardless of the monitor value and saving mode
        else:
            (self.model).save_weights(f"{self.directory}/checkpoint_{index:03d}")


class EvaluationCallback(tf.keras.callbacks.Callback):
    def __init__(self, interval, data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interval = interval
        self.data = data


    def on_epoch_end(self, epoch, logs = None):
        if ((epoch + 1) < self.interval) or ((epoch + 1) % self.interval):
            return

        predict(
            data = self.data, 
            model = self.model, 
            path = f"./predictions/training_{(epoch + 1):02d}"
        )


def generate_callbacks(interval, data):

    checkpoints = CheckpointCallback(
        directory = "./checkpoints",
        frequency = "epoch", 
        optimize = False,
        interval = 5,
        monitor = "accuracy",
        mode = "max"
    )

    logs = tf.keras.callbacks.TensorBoard(
        log_dir = f"logs/fit/{datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}",
        histogram_freq = 1
    )

    evaluation = EvaluationCallback(interval = interval, data = data)

    return [checkpoints, logs, evaluation]


def generate_model(input_shape: list[int], output_channels: int):
    """
    Generate a final model for the given arguments

    Arguments:
            input_shape (list): Input shape of images into the model
            output_channels (int): The amount of possible labels

    Returns:
            A tf.keras.Model generated from the given data 

    Raises:
            N/A
    """

    # Define the base model as MobileNetV2 without
    # it's top (e.g. classification) layers
    base_model = tf.keras.applications.MobileNetV2(
        input_shape  = input_shape, 
        include_top = False
    )

    # Define which layer activations to use
    layer_names = [
        'block_1_expand_relu',
        'block_3_expand_relu',
        'block_6_expand_relu',
        'block_13_expand_relu',
        'block_16_project',
    ]

    # Define the output of the base model
    base_model_outputs = [
        base_model.get_layer(name).output 
        for name in layer_names
    ]

    # Define the Downsampling
    downsampling_stack = tf.keras.Model(
        inputs = base_model.input, 
        outputs = base_model_outputs
    )
    downsampling_stack.trainable = False

    # Define the Upsampling
    upsampling_stack = [
        pix2pix.upsample(512, 3),
        pix2pix.upsample(256, 3),
        pix2pix.upsample(128, 3),
        pix2pix.upsample(64, 3),
    ]

    # Define the inputs of the model
    inputs = tf.keras.layers.Input(shape = input_shape)

    # Downsampling through the model
    skips = downsampling_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for upsample, skip in zip(upsampling_stack, skips):
        x = upsample(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    # Define output layer of the model
    outputs = tf.keras.layers.Conv2DTranspose(
        filters = output_channels, 
        kernel_size = 3, 
        strides = 2,
        padding = 'same'
    )
    x = outputs(x)

    return tf.keras.Model(inputs = inputs, outputs = x)


class UNet(tf.keras.Model):
    def __init__(self,  input_shape, output_channels, *args, **kwargs):
        super(UNet, self).__init__(*args, **kwargs)

        # Build the model pipeline for input and output
        self.pipeline = generate_model(
            input_shape = input_shape,
            output_channels = output_channels 
        )

        # Initialize the two metrics for evaluating the model
        self.scce = tf.keras.metrics.SparseCategoricalCrossentropy(from_logits = True, name = "loss")
        self.accuracy = tf.keras.metrics.Accuracy(name = "accuracy")


    # Define the metrics for evaluating the model
    @property
    def metrics(self):
        return [self.scce, self.accuracy]


    # Define the call function which is used on model(x)
    @tf.function
    def call(self, x, training = False):
        return self.pipeline(x)


    # Define prediction function used on model.predict(x)
    @tf.function
    def predict(self, x):
        return self.pipeline(x)


    # Define the training step
    @tf.function
    def train_step(self, data):

        # Take in the current set of data
        x, y = data

        # Pass the data through the model and update/calculate
        # the metrics for model evaluation as required
        with tf.GradientTape() as tape:
            prediction = self(x, training = True)
            step_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)(y, prediction)
            self.scce.update_state(y, prediction)
            self.accuracy.update_state(y, tf.argmax(prediction, axis = -1))

        # Calculate and apply the gradients
        gradients = tape.gradient(step_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        # Return the updated metrics for this training step
        return {metric.name: metric.result() for metric in self.metrics}


    # Define the testing step
    @tf.function
    def test_step(self, data):

        # Take in the current set of data
        x, y = data

        # Pass the data through the model and update the testing metrics
        with tf.GradientTape() as tape:
            prediction = self(x, training = False)
            self.SCCE_loss.update_state(y, prediction)
            self.SCCE_accy.update_state(y, prediction)

        # Return the losses for this testing step
        return {metric.name: metric.result() for metric in self.metrics}