import argparse
import json
from .data_loader import image_segmentation_generator, \
    verify_segmentation_dataset, ImageSegmentationGen
import os
import glob
import six
import matplotlib.pyplot as plt
from keras import optimizers
from keras import backend as K
import keras 
import tensorflow as tf
from typing import Union
import numpy as np
from typing import Callable

def gen_dice(y_true, y_pred, eps=1e-6):
    """both tensors are [b, h, w, classes] and y_pred is in logit form"""

    # [b, h, w, classes]
    pred_tensor = tf.nn.softmax(y_pred)
    y_true_shape = tf.shape(y_true)
    y_pred_shape = tf.shape(y_pred)
    pred_tensor_shape = tf.shape(pred_tensor)

    counts = tf.reduce_sum(y_true, axis=1)
    weights = 1 / (counts ** 2)
    weights = tf.where(tf.math.is_finite(weights), weights, eps)

    multed = tf.reduce_sum(y_true * y_pred, axis=1)
    summed = tf.reduce_sum(y_true + y_pred, axis=1)

    numerators = 2 * tf.reduce_sum(weights*multed, axis=-1)
    denom = tf.reduce_sum(weights*summed, axis=-1)
    dices = 1 - numerators / denom
    dices = tf.where(tf.math.is_finite(dices), dices, tf.zeros_like(dices))
    return tf.reduce_mean(dices)

def tversky_loss(y_true, y_pred):
    beta = 0.6
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = y_true * y_pred
    denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)
    loss = 1 - tf.reduce_sum(numerator) / tf.reduce_sum(denominator)
    return loss

  
def find_latest_checkpoint(checkpoints_path, fail_safe=True):

    def get_epoch_number_from_path(path):
        return path.replace(checkpoints_path, "").strip(".")

    # Get all matching files
    all_checkpoint_files = glob.glob(checkpoints_path + ".*")
    # Filter out entries where the epoc_number part is pure number
    all_checkpoint_files = list(filter(lambda f: get_epoch_number_from_path(f).isdigit(), all_checkpoint_files))
    if not len(all_checkpoint_files):
        # The glob list is empty, don't have a checkpoints_path
        if not fail_safe:
            raise ValueError("Checkpoint path {0} invalid".format(checkpoints_path))
        else:
            return None

    # Find the checkpoint file with the maximum epoch
    latest_epoch_checkpoint = max(all_checkpoint_files, key=lambda f: int(get_epoch_number_from_path(f)))
    return latest_epoch_checkpoint


def train(model,
          train_images,
          train_annotations,
          input_height=None,
          input_width=None,
          n_classes=None,
          verify_dataset=True,
          checkpoints_path=None,
          epochs=None,
          batch_size=5,
          validate=False,
          val_images=None,
          val_annotations=None,
          val_batch_size=5,
          auto_resume_checkpoint=False,
          load_weights=None,
          steps_per_epoch=256,
          optimizer_name=None, 
		  do_augment=False, 
		  classifier=None,
          loss_name=None
          ):


    n_classes = model.n_classes
    input_height = model.input_height
    input_width = model.input_width
    output_height = model.output_height
    output_width = model.output_width

    if validate:
        assert val_images is not None
        assert val_annotations is not None
#loss="categorical_crossentropy",                                               ##For CCE
#loss= lambda yTrue, yPred: tversky_loss(yTrue, yPred),                         ##For TL
#loss= lambda yTrue, yPred: gen_dice(yTrue, yPred),                             ##For DSC
    if optimizer_name is not None:
        model.compile(loss= lambda yTrue, yPred: gen_dice(yTrue, yPred),  
                      optimizer=optimizer_name,
                      metrics=['accuracy'])

    if checkpoints_path is not None:
        with open(checkpoints_path+"_config.json", "w") as f:
            json.dump({
                "model_class": model.model_name,
                "n_classes": n_classes,
                "input_height": input_height,
                "input_width": input_width,
                "output_height": output_height,
                "output_width": output_width
            }, f)

    if load_weights is not None and len(load_weights) > 0:
        print("Loading weights from ", load_weights)
        model.load_weights(load_weights)

    if auto_resume_checkpoint and (checkpoints_path is not None):
        latest_checkpoint = find_latest_checkpoint(checkpoints_path)
        if latest_checkpoint is not None:
            print("Loading the weights from latest checkpoint ",
                  latest_checkpoint)
            model.load_weights(latest_checkpoint)

    if verify_dataset:
        print("Verifying training dataset")
        verified = verify_segmentation_dataset(train_images, train_annotations, n_classes)
        assert verified
        if validate:
            print("Verifying validation dataset")
            verified = verify_segmentation_dataset(val_images, val_annotations, n_classes)
            assert verified

    train_gen = ImageSegmentationGen(
        train_images, train_annotations,  batch_size,  n_classes,
        input_height, input_width, output_height, output_width , do_augment=do_augment )

    if validate:
        val_gen = ImageSegmentationGen(
            val_images, val_annotations,  val_batch_size,
            n_classes, input_height, input_width, output_height, output_width)

    if not validate:
        for ep in range(epochs):
            print("Starting Epoch ", ep)
            history = model.fit_generator(train_gen, steps_per_epoch, epochs=epochs, workers=1)
            if checkpoints_path is not None:
                model.save_weights(checkpoints_path + "." + str(ep))
                print("saved ", checkpoints_path + ".model." + str(ep))		  
            
    else:
        for ep in range(epochs):
            print("Starting Epoch ", ep)
            history = model.fit_generator(train_gen, steps_per_epoch,
                                validation_data=val_gen,
                                validation_steps=200,  epochs=epochs, workers=1)
           
            if checkpoints_path is not None:
                model.save_weights(checkpoints_path + "." + str(ep))
                print("saved ", checkpoints_path + ".model." + str(ep))
            print("Finished Epoch", ep)  
            return history