##############################################################################
# Import Libraries
###############################################################################
import os
import numpy as np
import scipy.io as sio

import tensorflow as tf
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
K.set_image_data_format('channels_last')

from keras.models import Model
from keras.layers import Input, Conv3D, Conv3DTranspose, Dropout, BatchNormalization, Add
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import Constant
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint

def execute(cmd):
	print(cmd)
	os.system(cmd)
    
start_filter_num = 32
filter_size = 3
stride_size = 2
dropout_ratio = 0.1
batchnormflag = 1
    
export_path = '/data02/user-storage/ziping/brain_SPECT_segm/data/DL_train/train_results_proposed/layer_6'   + '_startfilter_' + str(start_filter_num) \
                                                                                                            + '_filtersize_' + str(filter_size) \
                                                                                                            + '_stride_' + str(stride_size) \
                                                                                                            + '_dropout_0dot' + str(int(10*dropout_ratio)) \
                                                                                                            + '_bn_' + str(batchnormflag) \
                                                                                                            + '_each' \
                                                                                                            + '_final_train'

if os.path.isdir(export_path):
	raise Exception('Trying to create an export directory that already exists!')
else:
	execute('mkdir ' + export_path)
    
###############################################################################
# Define loss and accuracy metrics
###############################################################################
def loss_fn(y_true, y_pred):

    y_true = tf.stop_gradient(y_true)
    
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true,logits=y_pred))

def dice_coef(y_true, y_pred):

    y_pred = tf.nn.softmax(y_pred)	
    y_pred = y_pred[:,:,:,:,0:6]
    y_true = y_true[:,:,:,:,0:6]
    
    TP = tf.reduce_sum(tf.minimum(y_pred,y_true),axis=[1,2,3])
    FP = tf.reduce_sum(tf.maximum(tf.subtract(y_pred,y_true),0),axis=[1,2,3])
    FN = tf.reduce_sum(tf.maximum(tf.subtract(y_true,y_pred),0),axis=[1,2,3])
    numerator= tf.multiply(TP,2.0)
    denominator = tf.add(tf.add(numerator,FP),FN)
    dice = tf.reduce_mean(tf.divide(numerator,denominator))
    
    return dice

def add_conv_layer(num_filter, filter_size, stride_size, input_layer, drop_prob, batchnorm_flag, bias_ct=0.03, leaky_alpha=0.01):
    layer = Conv3D(num_filter, (filter_size, filter_size, filter_size), # num. of filters and kernel size 
                   strides=stride_size,
                   padding='same',
                   use_bias=True,
                   kernel_initializer='glorot_normal', # Xavier init
                   bias_initializer=Constant(value=bias_ct))(input_layer)
    if batchnorm_flag == 1:
        layer = BatchNormalization(axis=-1)(layer)  
    layer = LeakyReLU(alpha=leaky_alpha)(layer) # activation func. 
    layer = Dropout(drop_prob)(layer) 

    return layer

def add_transposed_conv_layer(num_filter, filter_size, stride_size, input_layer, drop_prob, batchnorm_flag, bias_ct=0.03, leaky_alpha=0.01):
    layer = Conv3DTranspose(num_filter, (filter_size, filter_size, filter_size), # num. of filters and kernel size 
                   strides=stride_size,
                   padding='same',
                   use_bias=True,
                   kernel_initializer='glorot_normal', # Xavier init
                   bias_initializer=Constant(value=bias_ct))(input_layer)
    if batchnorm_flag == 1:
        layer = BatchNormalization(axis=-1)(layer)  
    layer = LeakyReLU(alpha=leaky_alpha)(layer) # activation func. 
    layer = Dropout(drop_prob)(layer)
    
    return layer

def nn_arch(start_filter_num, filter_size, stride_size, dropout_ratio, batchnormflag, input_size=(128,128,128,1)):
    
    input = Input(input_size) 

    # Encoder
    conv1 = add_conv_layer(start_filter_num, filter_size, (1, 1, 1), input, dropout_ratio, batchnormflag)                                                  # 128                  
    down1 = add_conv_layer(start_filter_num, filter_size, (stride_size, stride_size, stride_size), conv1, dropout_ratio, batchnormflag)                    # 64  

    conv2 = add_conv_layer(start_filter_num*2, filter_size, (1, 1, 1), down1, dropout_ratio, batchnormflag)                                                # 64
    down2 = add_conv_layer(start_filter_num*2, filter_size, (stride_size, stride_size, stride_size), conv2, dropout_ratio, batchnormflag)                  # 32
                           
    conv3 = add_conv_layer(start_filter_num*4, filter_size, (1, 1, 1), down2, dropout_ratio, batchnormflag)                                                # 32
    down3 = add_conv_layer(start_filter_num*4, filter_size, (stride_size, stride_size, stride_size), conv3, dropout_ratio, batchnormflag)                  # 16
    
    conv4 = add_conv_layer(start_filter_num*8, filter_size, (1, 1, 1), down3, dropout_ratio, batchnormflag)                                                # 16
    
    # Decoder
    up7 = add_transposed_conv_layer(start_filter_num*4, filter_size, (stride_size, stride_size, stride_size), conv4, dropout_ratio, batchnormflag)         # 32
    up7 = Add()([up7, conv3])                                                                                                                              # 32
    conv7 = add_conv_layer(start_filter_num*4, filter_size, (1, 1, 1), up7, dropout_ratio, batchnormflag)                                                  # 32
    
    up8 = add_transposed_conv_layer(start_filter_num*2, filter_size, (stride_size, stride_size, stride_size), conv7, dropout_ratio, batchnormflag)         # 64
    up8 = Add()([up8, conv2])                                                                                                                              # 64
    conv8 = add_conv_layer(start_filter_num*2, filter_size, (1, 1, 1), up8, dropout_ratio, batchnormflag)                                                  # 64
    
    up9 = add_transposed_conv_layer(start_filter_num, filter_size, (stride_size, stride_size, stride_size), conv8, dropout_ratio, batchnormflag)           # 128
    up9 = Add()([up9, conv1])                                                                                                                              # 128
    conv9 = add_conv_layer(start_filter_num, filter_size, (1, 1, 1), up9, dropout_ratio, batchnormflag)                                                    # 128
    
    output = add_conv_layer(7, 1, (1, 1, 1), conv9, 0, 0) 
    
    model = Model(inputs=[input], outputs = [output])
    
    return model

#----------------------------------------------------------
# Step-1: Importing train/val datasets
#----------------------------------------------------------
num_train = 480
X_train = np.zeros((num_train,128,128,128,1))
Y_train = np.zeros((num_train,128,128,128,7))

data_folder = '/data02/user-storage/ziping/brain_SPECT_segm/data/simul/'
label_folder = '/data02/user-storage/ziping/brain_SPECT_segm/data/ground_truth/cont/'

for i in range(num_train):
    
    pat_index = i+1
    
    print("Loading training data at pat index = " + str(pat_index) + " (Progress: " + str(i+1) + " / " + str(num_train) + ") .........")
    
    sample = np.fromfile(data_folder+str(pat_index)+'/dmip_recon/output/dmip_rec.bin',dtype='float32')
    sample = np.reshape(sample,[128,128,128],order='F')
	  
    label = np.fromfile(label_folder+'true_'+str(pat_index)+'.bin',dtype='float32')
    label = np.reshape(label,[128,128,128,7],order='F')	
	
    sample[np.isnan(sample)] = 0
    label[np.isnan(label)] = 0
    
    assert not np.any(np.isnan(sample))
    assert not np.any(np.isnan(label))
    
    if batchnormflag == 1:
        sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample))
	
    X_train[i,:,:,:,0] = sample
    Y_train[i,:,:,:,:] = label

    if (i==120):
        sio.savemat(export_path+'/train_sample_'+str(i)+'.mat',{'sample':sample})
        sio.savemat(export_path+'/train_label_'+str(i)+'.mat',{'label':label})

print("-------------------Training sample size---------------------")
print(X_train.shape)
print("--------------------Training label size---------------------")
print(Y_train.shape)

num_val = 100
X_val = np.zeros((num_val,128,128,128,1))
Y_val = np.zeros((num_val,128,128,128,7))

for i in range(num_val):
    
    pat_index = i+481
    
    print("Loading validation data at pat index = " + str(pat_index) + " (Progress: " + str(i+1) + " / " + str(num_val) + ") .........")
    
    sample = np.fromfile(data_folder+str(pat_index)+'/dmip_recon/output/dmip_rec.bin',dtype='float32')
    sample = np.reshape(sample,[128,128,128],order='F')
	  
    label = np.fromfile(label_folder+'true_'+str(pat_index)+'.bin',dtype='float32')
    label = np.reshape(label,[128,128,128,7],order='F')	
	
    sample[np.isnan(sample)] = 0
    label[np.isnan(label)] = 0
    
    assert not np.any(np.isnan(sample))
    assert not np.any(np.isnan(label))
    
    if batchnormflag == 1:
        sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample))
	
    X_val[i,:,:,:,0] = sample
    Y_val[i,:,:,:,:] = label

    if (i==64):
        sio.savemat(export_path+'/val_sample_'+str(i)+'.mat',{'sample':sample})
        sio.savemat(export_path+'/val_label_'+str(i)+'.mat',{'label':label})

print("-------------------Validation sample size---------------------")
print(X_val.shape)
print("--------------------Validation label size---------------------")
print(Y_val.shape)

#----------------------------------------------------------
# Step-2: Build and Train Model
#----------------------------------------------------------
num_GPUs = 2
config = tf.ConfigProto(device_count = {'GPU': num_GPUs})
sess = tf.Session(config=config)
K.tensorflow_backend.set_session(sess)

model = nn_arch(start_filter_num, filter_size, stride_size, dropout_ratio, batchnormflag)
parallel_model = multi_gpu_model(model, num_GPUs)
parallel_model.compile(loss = loss_fn, optimizer = 'adam', metrics = [dice_coef])

checkpoint = ModelCheckpoint(export_path + '/ep{epoch:03d}.hdf5', monitor='loss', verbose=1, save_best_only=False, mode='auto', period=10)
train_history = parallel_model.fit(x = X_train, y = Y_train, batch_size = 4, epochs = 250, verbose = 1, callbacks = [checkpoint], validation_data = (X_val, Y_val), shuffle=True)

#-----------------------------------------------------------
# Step-3: Save 
#----------------------------------------------------------	
model.save(export_path + '/final.hdf5')

# plot the loss function progression during training
loss = train_history.history['loss']
val_loss = train_history.history['val_loss']
train_acc = train_history.history['dice_coef']
val_acc = train_history.history['val_dice_coef']

# Save datasets to a matfile to open later in matlab
mdict = {"train_loss": loss, "val_loss": val_loss,"train_acc":train_acc,"val_acc":val_acc}
sio.savemat(export_path + '/stats.mat', mdict)