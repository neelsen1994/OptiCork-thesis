"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Based on the work of Waleed Abdulla (Matterport)
Modified by github.com/GustavZ

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last

    # OWN TRAINING START
    python coco.py train --model=imagenet --classes='person'
"""
# py2/3 compability
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import json
import skimage.draw
import time
import numpy as np
import imgaug  # https://github.com/aleju/imgaug (pip3 install imageaug)

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
#from pycocotools.coco import COCO
#from pycocotools.cocoeval import COCOeval
#from pycocotools import mask as maskUtils

import zipfile
from six.moves import urllib # py2/3 compability
import shutil

# Root directory of the project
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mmrcnn.config import Config
from mmrcnn import model as modellib,utils

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_WEIGHTS_DIR = os.path.join(ROOT_DIR, "weights")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_DIR = os.path.join(ROOT_DIR, "data/cork")

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(DEFAULT_WEIGHTS_DIR, "mobile_mask_rcnn_coco.h5")

############################################################
#  Configurations
############################################################


class CorkConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    ## Give the configuration a recognizable name
    NAME = "cork"

    ## GPU
    IMAGES_PER_GPU = 1
    GPU_COUNT = 2

    ## Number of classes (including background)
    NUM_CLASSES = 1 + 1

    ## Backbone Architecture
    BACKBONE = "mobilenetv1"
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    ## Resolution
    RES_FACTOR = 2
    IMAGE_MAX_DIM = 1024 // RES_FACTOR
    RPN_ANCHOR_SCALES = tuple(np.divide((32, 64, 128, 256, 512),RES_FACTOR))

    ## Losses
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }

    ## Steps
    STEPS_PER_EPOCH = 10000
    VALIDATION_STEPS = 50

    ## Additions
    TRAIN_BN = True
    POST_NMS_ROIS_INFERENCE = 100


############################################################
#  Dataset
############################################################

class CorkDataset(utils.Dataset):

    def load_cork(self, dataset_dir, subset):
        """Load a subset of the Cork dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("Wood", 1, "Wood")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        if subset == 'train':
            print("Training Data doesn't require empty annotations ------->")
            annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "Wood",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a cork dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "Wood":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "Wood":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)




############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on MS COCO")
    parser.add_argument('--dataset', required=False,
                        default = DEFAULT_DATASET_DIR,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    parser.add_argument('--classes', required=False,
                        default=None,
                        metavar="<class names>",
                        help='classes that should be trained on. Default are all')
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    print("Classes (None means all):", args.classes)

    # classes must be a list
    args.classes = list(args.classes)

    # Configurations
    if args.command == "train":
        config = CorkConfig()
    else:
        class InferenceConfig(CorkConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()[1]
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    print("> Loading weights from {} ".format(model_path))
    model.load_weights(model_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        # Training dataset.
        dataset_train = CorkDataset()
        dataset_train.load_cork(args.dataset, "train")
        dataset_train.prepare()

        # Validation dataset
        dataset_val = CorkDataset()
        dataset_val.load_cork(args.dataset, "val")
        dataset_val.prepare()

        print("Dataset information: Train")
        print("Image Count: {}".format(len(dataset_train.image_ids)))
        print("Class Count: {}".format(dataset_train.num_classes))
        for i, info in enumerate(dataset_train.class_info):
            print("{:3}. {:50}".format(i, info['name']))

        print(dataset_train.class_info[1]['source'])
    
        print("Dataset information: Val")
        print("Image Count: {}".format(len(dataset_val.image_ids)))
        print("Class Count: {}".format(dataset_val.num_classes))
        for i, info in enumerate(dataset_val.class_info):
            print("{:3}. {:50}".format(i, info['name']))

        print(dataset_val.class_info[1]['source'])

        # Image Augmentation
        # Right/Left flip 50% of the time
        augmentation = imgaug.augmenters.Fliplr(0.5)

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("> Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads',
                    augmentation=augmentation)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("> Fine tune {} stage 4 and up".format(config.BACKBONE))
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=120,
                    layers='4+',
                    augmentation=augmentation)

        # Training - Stage 3
        # Fine tune all layers
        print("> Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=160,
                    layers='all',
                    augmentation=augmentation)

    
    else:
        print("> '{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
