# object-detection-tensorflow-api

Experimentation with the Tensorflow object detection API.

## Setup

First, run `poetry install` to set up the virtual environment with all it's dependencies

then, run `poetry shell` to launch into the virtual environment

then, cd into models/research and run `pip install .` to install the Tensorflow object detection API

## Maintainance notes

When updating the object detection API, models/research/setup.py is from models/research/object_detection/packages/tf2/setup.py, so you will need to cd into models/research and do `cp object_detection/packages/tf2/setup.py .`

Protobuf is used to build the object_detection/protos/*.proto files. When they are updated, run `protoc object_detection/protos/*.proto --python_out=.` from the models/research folder.

To test the setup and/or maintenance environment and ensure that it is properly set up, run `python object_detection/builders/model_builder_tf2_test.py` from the models/research directory.

## Notes

### Split dataset into training and test images

To split the dataset use `split-folders`. Example:

```zsh
#                  train/val/test
split-folders --ratio .8 .1 .1 --output dataset/training -- dataset/original-images
```

also note that you have to run this command in the virtual environment (`poetry shell`) or else the program split-folders wouldn't be found.

### Models used

All the models that are used in this repository are located [here](https://drive.google.com/drive/folders/14puaQ0piRGmT_0ECUccETTk5fxcXEnRP?usp=sharing).

### How to train a model

#### Preparing training data

1. Gather all the images that you need and place them in the dataset/original-images folder.

2. Label **all** the images with LabelImg.

3. With all the annotation XML files generated for each image with LabelImg, convert the XML files to a single CSV file using scripts/xml_to_csv.py.

<!-- 4. Split the images into 2 folders, train and validation with a 80% and 20% ratio respectively, see [Split dataset into training and test images](#split-dataset-into-training-and-test-images) -->

4. Use scripts/split_csv.py to split the CSV file into a training and evaluation file, let's call them `train.csv` and `eval.csv`. Images will be selected randomly and there are options to stratify examples by class, making sure that objects from all classes are present in both datasets. The usual proportions are 75% to 80% of the annotated objects used for training and the rest for the evaluation dataset.

5. Create a "label map" for the classes using scripts/generate_pbtxt.py.

6. Convert each of your CSV files into two TFRecord files (eg. train.record and eval.record), a serialized data format that TensorFlow is most familiar with.

#### Training the model

1. Download your the neural network model of choice from either the Detection Model Zoo [\[TF1\]](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md)[\[TF2\]](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) or from the models trained for classification available [here](https://github.com/tensorflow/models/tree/master/research/slim#Pretrained) and [here](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet#pretrained-models). This is the step in which your choice of TensorFlow version will make a difference. From my experience, many of the classification models work with TF 1.15, but I am not aware if they work with TF 2.

2. Provide a training pipeline, which is a file with `.config` extension that describes the training procedure. The models provided in the Detection Zoo come with their own pipelines inside their `.tar.gz` file, but the classification models do not. In this situation, your options are to:
    
    -   download one that is close enough from [here](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs) (I have succesfully done that to train classification MobileNets V1, V2 and V3 for detection).
    -   create your own, by following [this tutorial](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md).
    
    The pipeline config file has some fields that must be adjusted before training is started. The first thing you'll definitely want to keep an eye on is the `num_classes` attribute, which you'll need to change to the number of classes in your personal dataset.
    
    Other importants fields are the ones with the `PATH_TO_BE_CONFIGURED` string. In these fields, you'll need to point to the files they ask for, such as the label map, the training and evaluation TFRecords and the neural network checkpoint, which is a file with an extension like `.ckpt` or `.ckpt.data-####-of-####`. This file also comes with the `.tar.gz` file.
    
    In case you are using a model from the Detection Zoo, set the `fine_tune_checkpoint_type` field to `"detection"`, otherwise, set it to `"classification"`.
    
    There are additional parameters that may affect how much RAM is consumed by the training process, as well as the quality of the training. Things like the batch size or how many batches TensorFlow can prefetch and keep in memory may considerably increase the amount of RAM necessary, but I won't go over those here as there is too much trial and error in adjusting those.

3. To train the model, cd into the object_detection folder and do `python ../scripts/model_main.py --pipeline_config_path=$PIPELINE_CONFIG_PATH --model_dir=$MODEL_DIR --alsologtostderr`

### Useful links

https://gist.github.com/douglasrizzo/c70e186678f126f1b9005ca83d8bd2ce \
https://github.com/tzutalin/labelImg \
https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9 \
https://medium.com/analytics-vidhya/detecting-custom-objects-on-video-stream-with-tensorflow-and-opencv-34406bd0ec9 \
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_training_and_evaluation.md \
https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10 \
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md
