import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix

import numpy as np

import datetime

import matplotlib.pyplot as plt


def draw_images(images, path):
    titles = ["Input Image", "True Mask", "Predicted Mask", "Segmentation"]

    # Clear and reset the figure
    plt.close('all')
    plt.figure()

    # For each image, plot it on the figure
    for index in range(len(images)):
        plt.subplot(1, len(images), index + 1)
        plt.title(titles[index])
        plt.imshow(tf.keras.utils.array_to_img(images[index]))
        plt.axis('off')

    # Save the figure
    plt.savefig(path)


def make_predictions(dataset, model, amount, path):
    dataset = (dataset.unbatch()).take(amount)

    # Iterate through each datapoint in 'dataset'
    for index, (image, mask) in enumerate(dataset):

        # Predict on the data 'image', and 
        # format the resulting output
        prediction = model(np.array([image]))
        prediction = tf.argmax(prediction, axis = -1)
        prediction = prediction[..., tf.newaxis][0]

        # Create the segmentation image from the 
        # mask and the original input image
        indices = prediction[:, :, 0] != 0
        segmentation = np.zeros(image.shape)
        segmentation[indices] = image[indices][:]

        # Draw the corrosponding set of images
        draw_images([image, mask, prediction, segmentation], f"{path}_{index:03d}.png")


class EvaluationCallback(tf.keras.callbacks.Callback):
    def __init__(self, interval, data, amount, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interval = interval
        self.data = data
        self.model = model
        self.amount = amount


    def on_epoch_end(self, epoch, logs = None):
        if ((epoch + 1) < self.interval) or ((epoch + 1) % self.interval):
            return

        make_predictions(
            dataset = self.data, 
            model = self.model, 
            amount = self.amount,
            path = f"./predictions/training_{(epoch + 1):02d}"
        )


def generate_callbacks(interval, data, model):

    checkpoints = tf.keras.callbacks.ModelCheckpoint(
        filepath = "./checkpoints/checkpoint-{epoch:03d}.ckpt",
        monitor ='accuracy',
        verbose = 1,
        save_best_only = False,
        save_weights_only = True, 
        mode = 'max', 
        save_freq = (len(data)*interval) + 1
    )

    logs = tf.keras.callbacks.TensorBoard(
        log_dir = f"logs/fit/{datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}",
        histogram_freq = 1
    )

    evaluation = EvaluationCallback(
        interval = interval,
        data = data, 
        model = model, 
        amount = 25
    )

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