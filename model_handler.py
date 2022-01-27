import datetime
import sys

import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix

import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt


def draw_images(image, mask, prediction, segmentation, path):
    """
    Draw the set of resulting prediction data onto a single image
    and save it to a specified file path

    Arguments:
            image (np.array): The original input image
            mask (np.array): The mask for the image
            prediction (np.array): The predicted segmentation of the image
            segmentation (np.array): The applied segmentation prediction image
            path (str): A full path to the desired output file including file name 

    Returns:
            N/A

    Raises:
            N/A
    """

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
    """
    Apply the segmentation 'prediction' to 'image'
    with an AND operation between the two

    Arguments:
            image (np.array): The original input image
            prediction (np.array): The predicted segmentation of the image

    Returns:
            A copy of the original image with the segmentation applied to it

    Raises:
            N/A
    """

    indices = prediction[:, :, 0] != 0
    segmentation = np.zeros(image.shape)
    segmentation[indices] = image[indices]
    return segmentation


def predict(data, model, path):
    """
    Predict on a flat array of images 'data' using the
    trained network 'model' and save the resulting output 
    to a specified 'path'

    This is typically reserved for smaller datasets to test
    on that do not require batching or any other special
    processing techniques applied to them

    Arguments:
            data (iterable): An iterable structure of datapoints
            model (tf.Keras.model): Model to infer on the data with
            path (str): Desired output directory path

    Returns:
            N/A

    Raises:
            N/A
    """

    # Iterate through each datapoint in 'data'
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


def batch_predict(data, model, path, verbose = True):
    """
    Predict on a batched set of images 'data' using the
    trained network 'model' and save the resulting output 
    to a specified 'path'

    This is typically used for larger datasets that require
    batching and use other special techniques for processing

    Arguments:
            data (tf.data.Dataset): Batched data containing images and masks
            model (tf.Keras.model): Model to infer on the data with
            path (str): Desired output directory path
            verbose (bool): Flag to toggle progress bar

    Returns:
            N/A

    Raises:
            N/A
    """

    # If required, setup a progress bar
    if verbose:
        l_bar = "{percentage:3.0f}%"
        m_bar = "{bar:20}"
        r_bar = "({n_fmt}/{total_fmt}) [{elapsed_s:01.0f}s < {remaining_s:01.0f}s] {rate_fmt}{postfix}"

        progress = tqdm(
            bar_format = f"{l_bar} |{m_bar}| {r_bar}",
            unit = "batch", 
            total = tf.data.experimental.cardinality(data).numpy(),
            ncols = 90
        )

    # Create a variable to track the amount of processed images
    # (NOTE: required because batching can have remainders)
    image_index = 0

    # Iterate through each batch of images and masks in 'data'
    for image_batch, mask_batch in data:

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
        if verbose:
            progress.update(1)


class CheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, directory, frequency, interval, optimize, verbose, metric, mode, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.directory = directory
        self.frequency = frequency
        self.interval = interval
        self.optimize = optimize
        self.verbose = verbose
        self.metric = metric
        self.mode = mode

        self.optimal_value = None
        self.progress = None


    def on_epoch_begin(self, epoch, logs = None):
        print(f"\nEpoch [{epoch + 1}/{self.params['epochs']}]", flush = True)

        if self.verbose:
            l_bar = "{percentage:3.0f}%"
            m_bar = "{bar:20}"
            r_bar = "({n_fmt}/{total_fmt}) [{elapsed_s:01.0f}s < {remaining_s:01.0f}s] {rate_fmt}{postfix}"

            self.progress = tqdm(
                bar_format = f"{l_bar} |{m_bar}| {r_bar}",
                unit = "steps", 
                total = self.params['steps'],
                ncols = 90
            )


    def on_epoch_end(self, epoch, logs = None):

        # Close the progress bar
        if self.verbose:
            self.progress.close()

        # Check for the checkpoint saving criteria being met
        if self.check_criteria("epoch", (epoch + 1), logs):
            (self.model).save_weights(f"{self.directory}/checkpoint-{(epoch + 1):03d}.ckpt")
            print(f"\nSaving checkpoint to '{self.directory}'...\n", flush = True)


    def on_train_batch_end(self, batch, logs = None):
        
        # Update the progress bar and it's displayed metrics
        if self.verbose:
            (self.progress).update(1)
            (self.progress).set_postfix(
                loss = f"{logs['loss']:.03f}", 
                accuracy = f"{logs['accuracy']:.03f}"
            )

        # Check for the checkpoint saving criteria being met
        if self.check_criteria("batch", (batch + 1), logs):
            (self.model).save_weights(f"{self.directory}/checkpoint-{(batch + 1):03d}.ckpt")
            print(f"\nSaving checkpoint to '{self.directory}'...\n", flush = True)


    def check_criteria(self, frequency, index, logs):

        # If the saving frequency isn't set to the given 'frequency'
        if self.frequency != frequency:
            return False

        # If the current 'index' doesn't match the saving interval
        if (index < self.interval) or (index % self.interval):
            return False

        # If the callback is set to save only the "best" results
        if self.optimize:

            # If this is the first time checking, save it as the initial result
            if self.optimal_value == None:
                self.optimal_value = logs[self.metric]
                return True

            # Else-if the qualification of "best" is maximization of the 
            # 'monitor' value and the current value meets that criteria
            elif self.mode == "max" and logs[self.metric] > self.optimal_value:
                self.optimal_value = logs[self.metric]
                return True

            # Else-if the qualification of "best" is minimization of the 
            # 'monitor' value and the current value meets that criteria
            elif self.mode == "min" and logs[self.metric] < self.optimal_value:
                self.optimal_value = logs[self.metric]
                return True

        # Else we save regardless of the monitor value and saving mode
        else:
            return True


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
    """
    Generate three seperate callbacks useful for fitting a model:

    - Model Checkpoint callback for saving model weights and monitoring
    - Tensorboard Logging callback for history checking & model comparison
    - Evaluation callback for manual mid-training evaluation of model performance

    Arguments:
            interval (int): The amount of iterations to wait before evaluating the
                            model and saving it's current set weights to a directory

            data (tf.data.Dataset): Dataset used to evaluate the model while training

    Returns:
            A single list containing each callback

    Raises:
            N/A
    """

    # Get a callback to save checkpoints while 
    # training the model to a given directory
    #
    # NOTE: Setting the steps-per-execution in compile()
    #       may break this when using verbose output, 
    #       and can be fixed by instead setting this 
    #       as a tf.keras.Callbacks.ModelCheckpoint
    #
    checkpoints = CheckpointCallback(
        directory = "./checkpoints",
        frequency = "epoch", 
        optimize = False,
        verbose = True,
        interval = interval,
        metric = "accuracy",
        mode = "max"
    )

    # Get a callback to save the history for
    # viewing within TensorBoard
    logs = tf.keras.callbacks.TensorBoard(
        log_dir = f"logs/fit/{datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}",
        histogram_freq = 1
    )

    # Get a callback for evaluating the model
    # on a dataset 'data' every 'interval' epochs
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