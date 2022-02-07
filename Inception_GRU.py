#SOFI-Mukhtar
# coding: utf-8

# In[2]:
!pip install tensorflow-gpu==1.15
!pip install keras==1.2.2
%cd '/content/drive/MyDrive/Colab Notebooks/Beta_turn'

# coding: utf-8

# In[3]:
import gzip
import h5py
import numpy as np
###########################################################################
import numpy as np
#file=open("/content/drive/MyDrive/Colab Notebooks/datasets/cb6133_avaialable.txt", 'r')
#lines=file.readlines()
#length=len(lines)
#x=0
#List=[]
#while x in range(length):
#  temp=lines[x]
 # temp=int(temp)
  #List.append(temp)
  #x+=1
#print(list)

############################################################################
cb6133 =  np.load(gzip.open('/content/drive/MyDrive/Colab Notebooks/datasets/bt4496_ss_g_t.npy.gz', 'rb'))

#cb6133=np.reshape(cb6133,(6133,700,57))
#print(cb6133.shape)
#labels = np.load(gzip.open('/content/drive/MyDrive/Colab Notebooks/datasets/bt4496_ss_g_t.npy.gz', 'rb'))
#labels=np.reshape(labels,(6133,700,3))
#print(labels.shape)
#cb6133=cb6133[List,:,:]
#labels=labels[List,:,:]
print(cb6133.shape)
#print(labels.shape)
# In[4]:
print("data loaded")
print(".................")
print("reshaping data")
#cb6133 = data= np.reshape(cb6133, (-1, 700, 79))
print("........................")
print("data reshape completed")

# In[5]:

#dataindex = ((range(22))+(range(31,33))+(range(35,57))
dataindex = list(range(46))+ list(range(68,76))+ list(range(46,68))
labelindex =list(range(76,79))
#solvindex = list(range(33,35))
maskindex = [30]
traindata = cb6133[:4000,:,dataindex]
trainlabel = cb6133[:4000,:,labelindex]
#trainsolvlabel = cb6133[:5600,:,solvindex]
#print(trainsolvlabel)
#trainsolvvalue = trainsolvlabel[:,:,0]*2 + trainsolvlabel[:,:,1]
#trainsolvlabel = np.zeros((trainsolvvalue.shape[0], trainsolvvalue.shape[1], 4))
"""
for i in range(trainsolvvalue.shape[0]):
    for j in range(trainsolvvalue.shape[1]):
        if np.sum(trainlabel[i,j,:]) != 0:
            p=int(trainsolvvalue[i,j])
            print(p)
            trainsolvlabel[i,j,int(trainsolvvalue[i,j])] = 1
"""
#trainmask = cb6133[:5260,:,maskindex]* -1 + 1 
valdata = cb6133[4000:4400,:,dataindex]
vallabel = cb6133[4000:4400,:,labelindex]
#valsolvlabel = cb6133[5600:5877,:,solvindex]
#valsolvvalue = valsolvlabel[:,:,0]*2 + valsolvlabel[:,:,1]
#valsolvlabel = np.zeros((valsolvvalue.shape[0], valsolvvalue.shape[1], 4))
"""
for i in range(valsolvvalue.shape[0]):
    for j in range(valsolvvalue.shape[1]):
        if np.sum(vallabel[i,j,:]) != 0:
            valsolvlabel[i,j,valsolvvalue[i,j]] = 1
"""
#valmask = cb6133[5260:5870,:,maskindex] * -1 + 1
#traindata = np.concatenate((traindata, valdata), axis=0)
traindataaux = traindata[:,:,0:54] #22
traindata = traindata[:,:,54:76]
#trainlabel = np.concatenate((trainlabel, vallabel), axis=0)
#trainsolvlabel = np.concatenate((trainsolvlabel, valsolvlabel), axis=0)
#trainmask = np.concatenate((trainmask, valmask), axis=0)
valdataaux = valdata[:,:,0:54] # 22
valdata = valdata[:,:,54:76]
testdata = cb6133[4400:4494,:,dataindex]
testlabel = cb6133[4400:4494,:,labelindex]

testdata1 = cb6133[4494:4496,:,dataindex]
testlabel1 = cb6133[4494:4496,:,labelindex]
#testsolvlabel = cb6133[5877:,:,solvindex]
#testsolvvalue = testsolvlabel[:,:,0]*2 + testsolvlabel[:,:,1]
#testsolvlabel = np.zeros((testsolvvalue.shape[0], testsolvvalue.shape[1], 4))
"""
for i in range(testsolvvalue.shape[0]):
    for j in range(testsolvvalue.shape[1]):
        if np.sum(testlabel[i,j,:]) != 0:
            testsolvlabel[i,j,testsolvvalue[i,j]] = 1
"""
#testmask = cb6133[5870:,:,maskindex] * -1 + 1
testdataaux = testdata[:,:,0:54] #22
testdata = testdata[:,:,54:76] #22

testdataaux1 = testdata1[:,:,0:54] #22
testdata1 = testdata1[:,:,54:76] #22
# convert one hot to interger
traindata = traindata[:,:,:21]
traindataint = np.ones((traindata.shape[0], traindata.shape[1]))
for i in range(traindata.shape[0]):
    for j in range(traindata.shape[1]):
        if np.sum(traindata[i,j,:]) != 0:
            traindataint[i,j] = np.argmax(traindata[i,j,:])
valdata = valdata[:,:,:21]
valdataint = np.ones((valdata.shape[0], valdata.shape[1]))
for i in range(valdata.shape[0]):
    for j in range(valdata.shape[1]):
        if np.sum(valdata[i,j,:]) != 0:
            valdataint[i,j] = np.argmax(valdata[i,j,:])
traindataint = np.concatenate((traindataint, valdataint), axis=0)
traindataaux = np.concatenate((traindataaux, valdataaux), axis=0)
#traindataaux[:,:,-1] = 1-traindataaux[:,:,-1]
print(traindataaux[7,0,:],traindataaux[7,699,:])
trainlabel = np.concatenate((trainlabel, vallabel), axis=0)
#trainsolvlabel = np.concatenate((trainsolvlabel, valsolvlabel), axis=0)
testdata = testdata[:,:,:21]
testdataint = np.ones((testdata.shape[0], testdata.shape[1]))
for i in range(testdata.shape[0]):
    for j in range(testdata.shape[1]):
        if np.sum(testdata[i,j,:]) != 0:
            testdataint[i,j] = np.argmax(testdata[i,j,:])

testdata1 = testdata1[:,:,:21]
testdataint1 = np.ones((testdata1.shape[0], testdata.shape[1]))
for i in range(testdata1.shape[0]):
    for j in range(testdata1.shape[1]):
        if np.sum(testdata1[i,j,:]) != 0:
            testdataint1[i,j] = np.argmax(testdata1[i,j,:])


def weighted_accuracy(y_true, y_pred):
    
    a=K.equal(K.argmax(y_true, axis=-1),K.argmax(y_pred, axis=-1))
    b=K.sum(y_true, axis=-1)
    c=K.sum(y_true)
    a=K.cast(a,K.floatx())
    b=K.cast(b,K.floatx())
    d=a*b
    e=d/c
    f=K.sum(e)
    return f
######################################################################
#FOCAL LOSS
def multi_category_focal_loss1(alpha, gamma=2.0):
    """
    focal loss for multi category of multi label problem
         Focal loss for multi-class or multi-label problems
         Alpha is used to specify the weight of different categories/tags. The array size needs to be the same as the number of categories.
         When there is a skew between different categories/tags in your dataset, try applying this function as a loss.
    Usage:
     model.compile(loss=[multi_category_focal_loss1(alpha=[1,2,3,2], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    epsilon = 1.e-7
    alpha = tf.constant(alpha, dtype=tf.float32)
    
    #alpha = tf.constant_initializer(alpha)
    gamma = float(gamma)
    def multi_category_focal_loss1_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
        ce = -tf.log(y_t)
        weight = tf.pow(tf.subtract(1., y_t), gamma)
        fl = tf.matmul(tf.multiply(weight, ce), alpha)
        loss = tf.reduce_mean(fl)
        return loss
    return multi_category_focal_loss1_fixed
######################################################################
from keras import backend as K


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss
#######################################################################
def convertPredictTurnLevelResult2HumanReadable(predictedGT):
    predictions = np.argmax(predictedGT, axis=-1)
    # convert back map meaning; 0 for Non-turn, 1 for turn, 2 for NoSeq, if any
    ssConvertMap = {0: 'n', 1: 't', 2: 'g' }
    result = []
    for i in range(len(predictions)):
        single=[]
        for j in range (0,700):
          #single.append
            single.append(ssConvertMap[predictions[i][j]])
        result.append(''.join(single))
    return result
######################################################################
# In[ ]:

#
import keras
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Input, Embedding, LSTM, Dense, merge, Convolution2D, Lambda, GRU, TimeDistributedDense, Reshape, Permute, Convolution1D, Masking
from keras.optimizers import Adam
from keras.regularizers import WeightRegularizer,l2
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping ,ModelCheckpoint

traindata.shape
#from loss_fun import multi_category_focal_loss1
#from loss_fun import multi_category_focal_loss1
#from loss import focal_loss

# In[ ]:
###################################################################################################
#Inception block
from keras.models import Model
from keras.layers import  BatchNormalization, Dropout
from keras.layers import Input,SpatialDropout1D, Embedding, LSTM, Dense, merge, Convolution2D, Lambda, GRU, TimeDistributed, Reshape, Permute, Convolution1D, Masking, Bidirectional
from keras.regularizers import l2
conv_layer_dropout_rate = 0.4
dense_layer_dropout_rate = 0.5

def inceptionBlock(x):
  x = BatchNormalization()(x)
  conv1_1 = Convolution1D(100, 1, activation='relu', border_mode='same', W_regularizer=l2(0.001))(x)
  conv1_1 = Dropout(conv_layer_dropout_rate)(conv1_1)
  conv1_1 = BatchNormalization()(conv1_1)

  conv2_1 = Convolution1D(100, 1, activation='relu', border_mode='same', W_regularizer=l2(0.001))(x)
  conv2_1 = Dropout(conv_layer_dropout_rate)(conv2_1)
  conv2_1 = BatchNormalization()(conv2_1)
  conv2_2 = Convolution1D(100, 3, activation='relu', border_mode='same', W_regularizer=l2(0.001))(conv2_1)
  conv2_2 = Dropout(conv_layer_dropout_rate)(conv2_2)
  conv2_2 = BatchNormalization()(conv2_2)

  conv3_1 = Convolution1D(100, 1, activation='relu', border_mode='same', W_regularizer=l2(0.001))(x)
  conv3_1 = Dropout(conv_layer_dropout_rate)(conv3_1)
  conv3_1 = BatchNormalization()(conv3_1)
  conv3_2 = Convolution1D(100, 3, activation='relu', border_mode='same', W_regularizer=l2(0.001))(conv3_1)
  conv3_2 = Dropout(conv_layer_dropout_rate)(conv3_2)
  conv3_2 = BatchNormalization()(conv3_2)
  conv3_3 = Convolution1D(100, 3, activation='relu', border_mode='same', W_regularizer=l2(0.001))(conv3_2)
  conv3_3 = Dropout(conv_layer_dropout_rate)(conv3_3)
  conv3_3 = BatchNormalization()(conv3_3)
  conv3_4 = Convolution1D(100, 3, activation='relu', border_mode='same', W_regularizer=l2(0.001))(conv3_3)
  conv3_4 = Dropout(conv_layer_dropout_rate)(conv3_4)
  conv3_4 = BatchNormalization()(conv3_4)

  concat = merge([conv1_1, conv2_2, conv3_4],mode='concat',concat_axis=-1)
  concat = BatchNormalization()(concat)

  return concat
########################################################################################################

main_input = Input(shape=(700,), dtype='int32', name='main_input')
#main_input = Masking(mask_value=23)(main_input)
x = Embedding(output_dim=50, input_dim=21, input_length=700)(main_input)
auxiliary_input = Input(shape=(700,54), name='aux_input')  #24
#auxiliary_input = Masking(mask_value=0)(auxiliary_input)
x = merge([x, auxiliary_input], mode='concat', concat_axis=-1)

block1_1 = inceptionBlock(x)

block2_1 = inceptionBlock(x)
block2_2 = inceptionBlock(block2_1)

block3_1 = inceptionBlock(x)
block3_2 = inceptionBlock(block3_1)
block3_3 = inceptionBlock(block3_2)
block3_4 = inceptionBlock(block3_3)

concat = merge([block1_1, block2_2, block3_4],mode='concat',concat_axis=-1)
concat = BatchNormalization()(concat)
#x = Reshape((1,700,74))(x)
#a = Convolution1D(64, 3, activation='relu', border_mode='same', W_regularizer=l2(0.001))(x)
#a = Permute((2,3,1))(a)
#b = Convolution1D(64, 7, activation='relu', border_mode='same', W_regularizer=l2(0.001))(x)
#b = Permute((2,3,1))(b)
#c = Convolution1D(64, 11, activation='relu', border_mode='same', W_regularizer=l2(0.001))(x)
#c = Permute((2,3,1))(c)
x = merge([concat,x], mode='concat', concat_axis=-1)
#x = Reshape((700,192))(x)
d = GRU(output_dim=300, return_sequences=True, activation='tanh', inner_activation='sigmoid', dropout_W=0.4)(x)
e = GRU(output_dim=300, return_sequences=True, activation='tanh', inner_activation='sigmoid', go_backwards=True, dropout_W=0.4)(x)
f = merge([d,e], mode='concat')
d = GRU(output_dim=300, return_sequences=True, activation='tanh', inner_activation='sigmoid', dropout_W=0.4)(f)
e = GRU(output_dim=300, return_sequences=True, activation='tanh', inner_activation='sigmoid', go_backwards=True, dropout_W=0.4)(f)
f = merge([d,e], mode='concat')
d = GRU(output_dim=300, return_sequences=True, activation='tanh', inner_activation='sigmoid', dropout_W=0.4)(f)
e = GRU(output_dim=300, return_sequences=True, activation='tanh', inner_activation='sigmoid', go_backwards=True, dropout_W=0.4)(f)
f = merge([d,e,x], mode='concat')
f = Dropout(0.4)(f)

f = Dense(200,activation='relu', W_regularizer=l2(0.002))(f)
#f = TimeDistributedDense(200,activation='relu', W_regularizer=l2(0.001))(f)
main_output = TimeDistributedDense(3,activation='softmax', name='main_output')(f)
#auxiliary_output = TimeDistributedDense(4,activation='softmax', name='aux_output')(f)
model = Model(input=[main_input, auxiliary_input], output=[main_output])
adam = Adam(lr=0.0005)
#alpha = np.array([[0.25],[0.25],[0.25]], dtype='float32')
weights = np.array([0.6, 0.9, 3.2])
model.compile(optimizer=adam,
              loss=weighted_categorical_crossentropy(weights),
              metrics=['accuracy', weighted_accuracy, 'precision', 'recall'])
model.summary()
#earlyStopping = EarlyStopping(monitor='val_acc', patience=8, verbose=1, mode='auto')
######################
earlyStopping = EarlyStopping(monitor='val_weighted_accuracy', patience=8, verbose=1, mode='auto')
load_file = "/content/drive/MyDrive/Colab Notebooks/DALSTM-master/DALSTM-master/DeepACLSTM/data/gru_bt6376_14_1_1.3_4.3_new_ss_inception.h5" # M: val_loss E: val_weighted_accuracy
checkpointer = ModelCheckpoint(filepath=load_file,verbose=1,save_best_only=True)

#################################
#load_file = "/content/drive/MyDrive/Colab Notebooks/DALSTM-master/DALSTM-master/DeepACLSTM/data/gru_bt6376_2.h5" # M: val_loss E: val_weighted_accuracy
#checkpointer = ModelCheckpoint(filepath=load_file,verbose=1,save_best_only=True)

#best_model_file = './ijcaibestacctwomasknowunoembedsoftmaxregular3e-4.h5' 
#best_model = ModelCheckpoint(load_file, monitor='val_weighted_accuracy', verbose=1, save_best_only = True) 
# and trained it via:
print(traindataint.shape, traindataaux.shape, trainlabel.shape, testdataint.shape, testdataaux.shape,testlabel.shape)
model.fit({'main_input': traindataint, 'aux_input': traindataaux},{'main_output': trainlabel},nb_epoch=200, batch_size=12, validation_data=({'main_input': valdataint, 'aux_input': valdataaux},{'main_output': vallabel}), callbacks=[earlyStopping,checkpointer], verbose=1,shuffle=True)
#model.fit({'main_input': traindataint, 'aux_input': traindataaux},{'main_output': trainlabel},nb_epoch=100, batch_size=96, validation_data=({'main_input': testdataint, 'aux_input': testdataaux},{'main_output': testlabel}), callbacks=[best_model], verbose=1)

model.load_weights(load_file)
print ("#########evaluate:##############")
#pip install 'h5py==2.10.0' --force-reinstall

#score = model.evaluate({'main_input': testdataint, 'aux_input': testdataaux},{'main_output': testlabel}, verbose=1, batch_size=2)
#print (score) 
#print ('test loss:', score[0])
#print ('test accuracy:', score[1])

predictions = model.predict({'main_input': testdataint1, 'aux_input': testdataaux1})
result=convertPredictTurnLevelResult2HumanReadable(predictions)
print("predicted turns")
print(result)
actual=convertPredictTurnLevelResult2HumanReadable(testlabel1)
print("actual turns")
print(actual)
# In[ ]:
