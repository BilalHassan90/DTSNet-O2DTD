import cv2
import os
import keras 
from keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from keras import backend as K
from functions.DTSNet import *
from keras.utils.vis_utils import plot_model
import tensorflow as tf
from keras import optimizers
import argparse
import json
from functions.data_loader import image_segmentation_generator, \
    verify_segmentation_dataset, ImageSegmentationGen
import glob
import six
from typing import Union
import numpy as np
from typing import Callable
from types import MethodType
from functions.train import train
from functions.predict import predict, predict_multiple, evaluate
from functions.data_loader import get_image_array, get_segmentation_array, DATA_LOADER_SEED, class_colors , get_pairs_from_paths
from tqdm import tqdm
import scipy.io


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
    beta = 0.5
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = y_true * y_pred
    denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)
    loss = 1 - tf.reduce_sum(numerator) / tf.reduce_sum(denominator)
    return loss



def evaluate( model=None , inp_images=None , annotations=None,inp_images_dir=None ,annotations_dir=None , checkpoints_path=None ):
    
    if model is None:
        assert (checkpoints_path is not None) , "Please provide the model or the checkpoints_path"
        model = model_from_checkpoint_path(checkpoints_path)
        
    if inp_images is None:
        assert (inp_images_dir is not None) , "Please privide inp_images or inp_images_dir"
        assert (annotations_dir is not None) , "Please privide inp_images or inp_images_dir"
        
        paths = get_pairs_from_paths(inp_images_dir , annotations_dir )
        paths = list(zip(*paths))
        inp_images = list(paths[0])
        annotations = list(paths[1])
        
    assert type(inp_images) is list
    assert type(annotations) is list
        
    tp = np.zeros( model.n_classes  )
    fp = np.zeros( model.n_classes  )
    fn = np.zeros( model.n_classes  )
    n_pixels = np.zeros( model.n_classes  )
    
    for inp , ann   in tqdm( zip( inp_images , annotations )):
        pr = predict(model , inp)
        # print("evaluate ", pr.shape)
        gt = get_segmentation_array( ann , model.n_classes ,  model.output_width , model.output_height , no_reshape=True  )
        gt = gt.argmax(-1)
        pr = pr.flatten()
        gt = gt.flatten()
                
        for cl_i in range(model.n_classes ):
            
            tp[ cl_i ] += np.sum( (pr == cl_i) * (gt == cl_i) )
            fp[ cl_i ] += np.sum( (pr == cl_i) * ((gt != cl_i)) )
            fn[ cl_i ] += np.sum( (pr != cl_i) * ((gt == cl_i)) )
            n_pixels[ cl_i ] += np.sum( gt == cl_i  )
            
    iou = tp / ( tp + fp + fn + 0.000000000001 ) 
    n_pixels_norm = n_pixels /  np.sum(n_pixels)
    f_weighted_iou = np.sum(iou*n_pixels_norm)
    mean_IoU = np.mean(iou)
    dice = 2 * mean_IoU / (mean_IoU + 1)
    print("Mean Dice Coefficient: ", dice)
    #Dict = {"frequency_weighted_IoU":f_weighted_iou , "mean_IoU":mean_IoU , "class_wise_IoU":iou}
    return f_weighted_iou, mean_IoU, iou

  

def predict(model=None, inp=None, out_fname=None, checkpoints_path=None):

    if model is None and (checkpoints_path is not None):
        model = model_from_checkpoint_path(checkpoints_path)

    assert (inp is not None)
    assert((type(inp) is np.ndarray) or isinstance(inp, six.string_types)
           ), "Inupt should be the CV image or the input file name"

    if isinstance(inp, six.string_types):
        inp = cv2.imread(inp)

    assert len(inp.shape) == 3, "Image should be h,w,3 "
    orininal_h = inp.shape[0]
    orininal_w = inp.shape[1]

    output_width = model.output_width
    output_height = model.output_height
    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes

    x = get_image_array(inp, input_width, input_height, ordering=IMAGE_ORDERING)
    # print(x.shape)
    pr = model.predict(np.array([x]))[0]
    # print(pr.shape)
    allScores = pr.reshape((output_height,  output_width, n_classes))
    # print("allScores", allScores)
    # print("Prnew", allScores.shape)
    # Score=tf.reduce_max(allScores, axis=-1)
    Score=np.max(allScores, axis=-1)
    pr = pr.reshape((output_height,  output_width, n_classes)).argmax(axis=2)
    Cr=pr
    # print("Score", Score.shape)
    # print("Score", Score)
    # print("Cr", Cr.shape)
    # print("Cr", Cr)

    seg_img = np.zeros((output_height, output_width, 3))
    CrVar = Cr.astype('uint8')
    ScoreVar = Score.astype('float')
    allScoresVar = allScores.astype('float')
    colors = class_colors

    for c in range(n_classes):
        seg_img[:, :, 0] += ((pr[:, :] == c)*(colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((pr[:, :] == c)*(colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((pr[:, :] == c)*(colors[c][2])).astype('uint8')

    seg_img = cv2.resize(seg_img, (orininal_w, orininal_h),interpolation = cv2.INTER_NEAREST)

    if out_fname is not None:
        cv2.imwrite(out_fname, seg_img)                  ##CommentifnoImagesReq
        out_fname=out_fname.replace("png","mat")
        scipy.io.savemat(out_fname, dict(C=CrVar, Score=ScoreVar,allScores=allScoresVar))

    return pr


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))

config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

model = DTSNet(n_classes=4,  height=576, width=768)

new_model = tf.keras.models.load_model('TrainedInstances/Model.h5', compile = False)

# new_model.compile(loss= "categorical_crossentropy")                           ## For CE
new_model.compile(loss= lambda yTrue, yPred: gen_dice(yTrue, yPred))            ## For DSC
# new_model.compile(loss= lambda yTrue, yPred: tversky_loss(yTrue, yPred))      ## For TL

new_model.n_classes = model.n_classes

new_model.output_width=model.output_width
new_model.output_height=model.output_height
new_model.n_classes=model.n_classes
new_model.input_height=model.input_height
new_model.input_width=model.input_width


folder = "Data/TestDataset/Images/"

f_weighted_iou, mean_IoU, iou = evaluate(new_model, inp_images_dir=folder  , 
	 annotations_dir="Data/TestDataset/Labels" )
print(iou)

print({"frequency_weighted_IoU":f_weighted_iou , "mean_IoU":mean_IoU , "class_wise_IoU":iou})

np.savetxt('Data/TestDataset/results_summary/results.csv', iou, fmt='%s')

for filename in os.listdir(folder):
    out = predict(new_model, inp=os.path.join(folder,filename), out_fname=os.path.join("Data/TestDataset/ResultsMATFiles/", filename))
   
