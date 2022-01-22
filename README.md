<h1 align="center">Tensorflow Image Segmentation</h1> 
    <p align="center">
        An alternative implementation of Tensorflow's Image Segmentation Guide
        <br/><br/><br/>
        [<a href="https://www.tensorflow.org/">Tensorflow</a>]
        [<a href="https://github.com/Haskili/TF_Segmentation#acknowledgements">Acknowledgements</a>]
        [<a href="https://github.com/Haskili/TF_Segmentation/issues">Issues</a>]
    </p>
</p>

## Overview
This is an alternative take on the implementation shown in Tensorflow's "[Image Segmentation](https://www.tensorflow.org/tutorials/images/segmentation)" guide. I wrote this primarily to illustrate how to perform similar techniques that lean more on the features within Tensorflow on real (e.g. [AIFM](https://aiformankind.com) & [HPWREN](http://hpwren.ucsd.edu)'s 'Wildfire') datasets. 

A few examples of features/implementations which were not otherwise covered in the original are sub-classing the model into it's own derived `tf.Keras.Model` class that can easily have it's weights saved and loaded into a new model, applying the segmentation output onto the images to better evaluate model performance, writing the process within the context of available popular online datasets and their respective formats, etc.

<br>
    <p align="center">
        <img src = "https://i.imgur.com/Q0DK7RU.gif" alt ="" width="50%" height="50%">
    </p>
<br>

While the current implementation remains limited given the initial use-case it was designed for, the current plan is to get more appropriate datasets to better demonstrate the utility and go from there.
<br></br>

## Requirements

**Libraries -- Tensorflow (2.7.0-1)**
<br></br>
Requires Tensorflow (2.7.0-1) or later, please see dependencies listed [here](https://archlinux.org/packages/community/x86_64/tensorflow/).
<br></br>

**Datasets & Formatting**
<br></br>
The current state of the project supports only two different dataset formats, the first of which being COCO JSON. From the COCO JSON format, it then transforms the dataset into the other format, an example-based CSV dataset file. This CSV format is designed so that it can be easily read in as a Tensorflow `tf.data.Dataset` object by one the functions within `dataset_handler.py`.

For a few ideas on where to start looking for more of these dataset, check popular dataset repositories such as the [Roboflow Object Detection Datasets](https://public.roboflow.com/object-detection). Robowflow, like many of the more prominent sites, even allows users to download datasets in alternative formats and with occasional augmentation.
<br></br>

## Dataset Loading & Generation

To begin, check out the first few lines of `train.py` for parsing the COCO JSON file:
```python
parse_coco_json(
    input_path = f"./{DATASET}/train/_annotations.coco.json", 
    output_path = f"./annotations_training.csv", 
    image_path = f"./{DATASET}/train", 
    labels = LABELS[DATASET]
)
```

With that, the next step is then generating and saving a mask for each image in the dataset using the CSV file we just created, 
```python
generate_dataset_masks(
    input_file = "./annotations_training.csv", 
    image_shape = DATASET_SIZE[DATASET],
    input_size = INPUT_SIZE
)
```

Afterwards, all that's left is to generate & augment the `tf.data.Dataset` object that's fed to the `model` during training/testing,
```python
dataset = generate_dataset("./annotations_training.csv", INPUT_SIZE) 
dataset_batches = dataset.map(
    preprocess_datapoint, 
    num_parallel_calls = tf.data.AUTOTUNE
)

dataset_batches = (
    dataset_batches
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .map(Augment(seed = 50, rotation = 0.035))
    .prefetch(buffer_size = tf.data.AUTOTUNE)
)
```

<br></br>

## Training

After reading in the `dataset` as shown in `train.py`, the next step is to initialize & compile a new `UNet` model like so:
```python
model = UNet(
    input_shape = [INPUT_SIZE, INPUT_SIZE, 3],
    output_channels = 2
)
model.compile(optimizer = 'adam')
```

Once that's done, the next (optional) step is to make a few example predictions to make sure everything is loaded in correctly and is functioning as expected.
```python
batch_predict(
    dataset = dataset_batches, 
    model = model, 
    path =  "./predictions/example"
)
```

The last step is to make a call to `fit()`, passing to it the training session information (e.g. `EPOCHS`) that it needs, as well as simultaneously generating callbacks for things such as checkpoint saving, logging, mid-training evaluation, etc.
```python
model.fit(
    dataset_batches, 
    epochs = EPOCHS,
    callbacks = generate_callbacks(
        interval = INTERVAL, 
        data = dataset_batches
    )
)
```

<br></br>

## Testing

For starters, simply load in the dataset that needs to tested (as in the previous sections), and initialize a new `UNet` model, similar to what's shown in `test.py`.
```python
model = UNet(
    input_shape = [INPUT_SIZE, INPUT_SIZE, 3],
    output_channels = 2
)
model.load_weights(f"./checkpoints/checkpoint-{CKPT_INDEX}.ckpt").expect_partial()
```

After that, the last step is to call `make_predictions()` to pass all the data to the model and parse the resulting output as needed.
```python
batch_predict(
    dataset = dataset_batches, 
    model = model,
    path = "./predictions/testing"
)
```
<br></br>

## Results
<br>
<p align="center">
    <img src = "https://i.imgur.com/i9vxStG.gif" alt ="" width="80%" height="80%"><br></br>
    <img src = "https://i.imgur.com/9hslh4P.gif" alt ="" width="80%" height="80%"><br></br>
    <img src = "https://i.imgur.com/s1Neay5.gif" alt ="" width="80%" height="80%"><br></br>
    <img src = "https://i.imgur.com/BaoJmXj.gif" alt ="" width="80%" height="80%"><br></br>
    <img src = "https://i.imgur.com/sf6pMNd.gif" alt ="" width="80%" height="80%"><br></br>
    <img src = "https://i.imgur.com/Q0DK7RU.gif" alt ="" width="80%" height="80%"><br></br>
</p>
<br></br>

## Acknowledgements

**ACKOWLEDGEMENTS**
- ...

<br>

**RESOURCES**

[Roboflow](https://public.roboflow.com/object-detection)
> Roboflow stands as a decent source for a number of unique datasets, and offers a multitude of options for augmentation have proven very helpful

<br>

[Name](site)
> ...

<br>

**MISC_INFO**
- ...