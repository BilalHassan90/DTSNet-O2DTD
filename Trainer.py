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



print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))

config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

model = DTSNet(n_classes=4,  height=576, width=768)

learning_rate=0.1			## (0.1, 0.01, 0.001)
optimizer= keras.optimizers.Adadelta(learning_rate)		## Adadelta    
## optimizer= keras.optimizers.SGM(learning_rate, momentum = 0.9)	## SGDM                  
## optimizer= keras.optimizers.Adam(learning_rate)			## SGDM                  

print("Learning rate is", learning_rate)

history = model.train(
    train_images =  "Data/TrainDataset/Images",
    train_annotations = "Data/TrainDataset/Labels/",
	val_images =  "Data/ValidationDataset/Images/",
    val_annotations = "Data/ValidationDataset/Labels/",
    checkpoints_path = None , epochs=20, validate=True, optimizer_name=optimizer, loss_name=None
)


print(history.history.keys())

f, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)
ax1.plot(history.history['accuracy'],'r')
ax1.plot(history.history['val_accuracy'],'g')
ax1.set_title('Model Accuracy')
ax1.set(xlabel='Epoch', ylabel='Accuracy')
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax1.legend(['Training', 'Validation'], loc='upper left')

ax2.plot(history.history['loss'],'r')
ax2.plot(history.history['val_loss'],'g')
ax2.set_title('Model Loss')
ax1.set(xlabel='Epoch', ylabel='Loss')
ax2.legend(['Training', 'Validation'], loc='upper left')
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
f.savefig('Training Graph/TrainingGraph.png')
f.show()
plt.close(f)

model.save("TrainedInstances/Model.h5")

f_weighted_iou, mean_IoU, iou = model.evaluate_segmentation(inp_images_dir="Data/TestDataset/Images/"  , 
	annotations_dir="Data/TestDataset/Labels/" )
print(iou)

print({"frequency_weighted_IoU":f_weighted_iou , "mean_IoU":mean_IoU , "class_wise_IoU":iou})



TATL=[history.history['accuracy'], history.history['val_accuracy'], history.history['loss'], history.history['val_loss'], iou]
np.savetxt('Model.csv', TATL, delimiter=',',fmt='%s')

folder = "Data/TestDataset/Images/"
for filename in os.listdir(folder):
    out = model.predict_segmentation(inp=os.path.join(folder,filename), out_fname=os.path.join("Data/TestDataset/segmentation_results/",filename))
    

