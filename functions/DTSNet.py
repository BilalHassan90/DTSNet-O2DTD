import numpy as np
import keras
from keras.models import *
from keras.models import Model
from keras.layers import *
import keras.backend as K
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
import tensorflow as tf
from .config import IMAGE_ORDERING
from types import MethodType
from .train import train
from .predict import predict, predict_multiple, evaluate
    
def DTSNet(n_classes,  height=576, width=768):

    modelSegmentor = get_segmentor(n_classes, height=height, width=width)
    modelSegmentor.model_name = "segmentor"
	
    return modelSegmentor
  
def get_segmentor(classes, height=576, width=768):

    assert height % 192 == 0
    assert width % 192 == 0

    inputLayer, features = encoder(height=height,  width=width)
        
    o = decoder(features, classes)
	
    modelSegmentor = get_segmentation_model(inputLayer, o)
    return modelSegmentor
    
def encoder(height,  width):
    from .config import IMAGE_ORDERING as order
    
    assert height > 0 == 0
    assert width > 0 == 0

    if order == 'channels_first':
        inputLayer = Input(shape=(3, height, width))
    elif order == 'channels_last':
        inputLayer = Input(shape=(height, width, 3))

    if order == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    kernel_size = 3
    
    x = ZeroPadding2D((3, 3), data_format=order)(inputLayer)
    x = Conv2D(64, (7, 7), data_format=order, strides=(2, 2))(x)
    s1 = x 
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), data_format=order, strides=(2, 2))(x)
    x_i = x
    x = Conv2D(64, (1, 1), data_format=order, strides=(1, 1))(x_i)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(64, kernel_size, data_format=order,padding='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (1, 1), data_format=order)(x)
    x = BatchNormalization(axis=bn_axis)(x)
    sc = Conv2D(256, (1, 1), data_format=order,strides=(1, 1))(x_i)
    sc = BatchNormalization(axis=bn_axis)(sc)
    x = layers.add([x, sc])
    x = Activation('relu')(x)
    x_i = x
    x = Conv2D(64, (1, 1), data_format=order)(x_i)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(64, kernel_size, data_format=order, padding='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (1, 1), data_format=order)(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = layers.add([x, x_i])
    x = Activation('relu')(x)
    x_i = x
    x = Conv2D(64, (1, 1), data_format=order)(x_i)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(64, kernel_size, data_format=order, padding='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (1, 1), data_format=order)(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = layers.add([x, x_i])
    x = Activation('relu')(x)
    s2 = ZeroPadding2D((1, 1), data_format=order)(x)
    if order == 'channels_first':
        s2 = Lambda(lambda s2: s2[:, :, :-1, :-1])(s2)
    elif order == 'channels_last':
        s2 = Lambda(lambda s2: s2[:, :-1, :-1, :])(s2)
    x_i = x
    x = MaxPooling2D((1, 1), data_format=order, strides=(2, 2))(x_i)#####Conv2D(128, (1, 1), data_format=order, strides=(2, 2))(x_i)
    x = Conv2D(128, (1, 1), data_format=order, strides=(1, 1))(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(128, kernel_size, data_format=order,padding='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (1, 1), data_format=order)(x)
    x = BatchNormalization(axis=bn_axis)(x)
    sc = Conv2D(512, (1, 1), data_format=order,strides=(2, 2))(x_i)
    sc = BatchNormalization(axis=bn_axis)(sc)
    x = layers.add([x, sc])
    x = Activation('relu')(x)    
    x_i = x
    ##ResBlock1
    x = Conv2D(128, (1, 1), data_format=order)(x_i)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(128, kernel_size, data_format=order, padding='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (1, 1), data_format=order)(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = layers.add([x, x_i])
    x = Activation('relu')(x)
    x_i = x
    ##ResBlock2
    x = Conv2D(128, (1, 1), data_format=order)(x_i)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(128, kernel_size, data_format=order, padding='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (1, 1), data_format=order)(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = layers.add([x, x_i])
    x = Activation('relu')(x)
    x_i = x
    ##ResBlock3
    x = Conv2D(128, (1, 1), data_format=order)(x_i)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(128, kernel_size, data_format=order, padding='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (1, 1), data_format=order)(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = layers.add([x, x_i])
    x = Activation('relu')(x)
    s3 = x
    x_i = x
    ##TPBlock
    x = MaxPooling2D((1, 1), data_format=order, strides=(2, 2))(x_i)    ######Conv2D(256, (1, 1), data_format=order, strides=(2, 2))(x_i)
    x = Conv2D(256, (1, 1), data_format=order, strides=(1, 1))(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(256, kernel_size, data_format=order,padding='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(1024, (1, 1), data_format=order)(x)
    x = BatchNormalization(axis=bn_axis)(x)
    sc = Conv2D(1024, (1, 1), data_format=order,strides=(2, 2))(x_i)
    sc = BatchNormalization(axis=bn_axis)(sc)
    x = layers.add([x, sc])
    x = Activation('relu')(x)
    x_i = x
    ##ResBlock1
    x = Conv2D(256, (1, 1), data_format=order)(x_i)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(256, kernel_size, data_format=order, padding='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(1024, (1, 1), data_format=order)(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = layers.add([x, x_i])
    x = Activation('relu')(x)
    x_i = x
    ##ResBlock2
    x = Conv2D(256, (1, 1), data_format=order)(x_i)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(256, kernel_size, data_format=order, padding='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(1024, (1, 1), data_format=order)(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = layers.add([x, x_i])
    x = Activation('relu')(x)
    x_i = x
    ##ResBlock3
    x = Conv2D(256, (1, 1), data_format=order)(x_i)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(256, kernel_size, data_format=order, padding='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(1024, (1, 1), data_format=order)(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = layers.add([x, x_i])
    x = Activation('relu')(x)
    x_i = x
    ##ResBlock4
    x = Conv2D(256, (1, 1), data_format=order)(x_i)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(256, kernel_size, data_format=order, padding='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(1024, (1, 1), data_format=order)(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = layers.add([x, x_i])
    x = Activation('relu')(x)
    x_i = x
    ##ResBlock5
#    x = Conv2D(256, (1, 1), data_format=order)(x_i)
#    x = BatchNormalization(axis=bn_axis)(x)
#    x = Activation('relu')(x)
#    x = Conv2D(256, kernel_size, data_format=order, padding='same')(x)
#    x = BatchNormalization(axis=bn_axis)(x)
#    x = Activation('relu')(x)
#    x = Conv2D(1024, (1, 1), data_format=order)(x)
#    x = BatchNormalization(axis=bn_axis)(x)
#    x = layers.add([x, x_i])
#    x = Activation('relu')(x)
    s4 = x
    x_i = x
    ##TPBlock
    x = MaxPooling2D((1, 1), data_format=order, strides=(2, 2))(x_i)    #####Conv2D(512, (1, 1), data_format=order, strides=(2, 2))(x_i)
    x = Conv2D(512, (1, 1), data_format=order, strides=(1, 1))(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(512, kernel_size, data_format=order,padding='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(2048, (1, 1), data_format=order)(x)
    x = BatchNormalization(axis=bn_axis)(x)
    sc = Conv2D(2048, (1, 1), data_format=order,strides=(2, 2))(x_i)
    sc = BatchNormalization(axis=bn_axis)(sc)
    x = layers.add([x, sc])
    x = Activation('relu')(x)
    x_i = x
    ##ResBlock1
    x = Conv2D(512, (1, 1), data_format=order)(x_i)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(512, kernel_size, data_format=order, padding='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(2048, (1, 1), data_format=order)(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = layers.add([x, x_i])
    x = Activation('relu')(x)
    x_i = x
    ##ResBlock2
    x = Conv2D(512, (1, 1), data_format=order)(x_i)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(512, kernel_size, data_format=order, padding='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(2048, (1, 1), data_format=order)(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = layers.add([x, x_i])
    x = Activation('relu')(x)
    s5 = x
    
    features = [s1, s2, s3, s4, s5]
        
        
    return inputLayer, features
    

def decoder(features, classes):
    from .config import IMAGE_ORDERING as order

    [first, second, third, forth, lastFeatures] = features
    pool_factors = [1, 2, 3, 6]
    list = [lastFeatures]
	
    for p in pool_factors:
        if order == 'channels_first':
            h = K.int_shape(lastFeatures)[2]
            w = K.int_shape(lastFeatures)[3]
        elif order == 'channels_last':
            h = K.int_shape(lastFeatures)[1]
            w = K.int_shape(lastFeatures)[2]

        pool_size = strides = [
            int(np.round(float(h) / p)),
            int(np.round(float(w) / p))]

        pooledResult = AveragePooling2D(pool_size, data_format=order,
                         strides=strides, padding='same')(lastFeatures)
        pooledResult = Conv2D(512, (1, 1), data_format=order,
               padding='same', use_bias=False)(pooledResult)
        pooledResult = BatchNormalization()(pooledResult)
        pooledResult = Activation('relu')(pooledResult)

        pooledResult = resize_image(pooledResult, strides, data_format=order)
        list.append(pooledResult)
		
    if order == 'channels_first':
        lastFeatures = Concatenate(axis=1)(list)
    elif order == 'channels_last':
        lastFeatures = Concatenate(axis=-1)(list)

    lastFeatures = Conv2D(512, (1, 1), data_format=order, use_bias=False)(lastFeatures)
    lastFeatures = BatchNormalization()(lastFeatures)
    lastFeatures = Activation('relu')(lastFeatures)
    
    f1 = Conv2D(1024, (1, 1), data_format=order, use_bias=False, padding='same')(lastFeatures)
    f1 = resize_image(f1, (2, 2), data_format=order)
    f1 = layers.add([f1, forth])
    f1 = Conv2D(512, (1, 1), data_format=order, use_bias=False)(f1)
    f1 = BatchNormalization()(f1)
    f1 = Activation('relu')(f1)
    
    f2 = resize_image(f1, (2, 2), data_format=order)
    f2 = layers.add([f2, third])
    f2 = Conv2D(512, (1, 1), data_format=order, use_bias=False)(f2)
    f2 = BatchNormalization()(f2)
    f2 = Activation('relu')(f2)
    
    f3 = Conv2D(256, (1, 1), data_format=order, use_bias=False, padding='same')(f2)
    f3 = resize_image(f3, (2, 2), data_format=order)
    f3 = layers.add([f3, second])
    f3 = Conv2D(512, (1, 1), data_format=order, use_bias=False)(f3)
    f3 = BatchNormalization()(f3)
    f3 = Activation('relu')(f3)    
    lastFeatures = f3
 
    lastFeatures = Conv2D(classes, (3, 3), data_format=order, padding='same')(lastFeatures)        
    lastFeatures = resize_image(lastFeatures, (2, 2), data_format=order)
        
    return lastFeatures
    
    
    
    
def get_segmentation_model(input, output):

    img_input = input
    o = output

    o_shape = Model(img_input, o).output_shape
    i_shape = Model(img_input, o).input_shape

    if IMAGE_ORDERING == 'channels_first':
        output_height = o_shape[2]
        output_width = o_shape[3]
        input_height = i_shape[2]
        input_width = i_shape[3]
        n_classes = o_shape[1]
        o = (Reshape((-1, output_height*output_width)))(o)
        o = (Permute((2, 1)))(o)
    elif IMAGE_ORDERING == 'channels_last':
        output_height = o_shape[1]
        output_width = o_shape[2]
        input_height = i_shape[1]
        input_width = i_shape[2]
        n_classes = o_shape[3]
        o = (Reshape((output_height*output_width, -1)))(o)
        o = (Activation('softmax'))(o)
        model = Model(img_input, o)
        model.output_width = output_width
        model.output_height = output_height
        model.n_classes = n_classes
        model.input_height = input_height
        model.input_width = input_width
        model.model_name = ""
        model.train = MethodType(train, model)
        model.predict_segmentation = MethodType(predict, model)
        model.predict_multiple = MethodType(predict_multiple, model)
        model.evaluate_segmentation = MethodType(evaluate, model)

    return model






def resize_image(inp,  s, data_format):

    try:

        return Lambda(lambda x: K.resize_images(x,
                                                height_factor=s[0],
                                                width_factor=s[1],
                                                data_format=data_format,
                                                interpolation='bilinear'))(inp)

    except Exception as e:
        # if keras is old, then rely on the tf function
        # Sorry theano/cntk users!!!
        assert data_format == 'channels_last'
        assert IMAGE_ORDERING == 'channels_last'

        import tensorflow as tf

        return Lambda(
            lambda x: tf.image.resize_images(
                x, (K.int_shape(x)[1]*s[0], K.int_shape(x)[2]*s[1]))
        )(inp)

