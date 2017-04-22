
# coding: utf-8

# In[1]:

import numpy as np
import math
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy.random import shuffle
from collections import deque
from scipy.stats import norm
import re
import os
from os import walk
import csv
import cv2
from moviepy.editor import *
from IPython.display import HTML
from sklearn.model_selection import train_test_split
from sklearn.utils import *
import sklearn

get_ipython().magic('matplotlib inline')


# In[2]:

log_path = './data/data'
data_df = pd.read_csv(log_path+'/driving_log.csv')
data_df['steering'].astype('float').plot.hist(bins=50)
data_df.iloc[:1]
data_df['steering'].describe()


# In[3]:

def select_data_set(data_df, num):
    steering_history = deque([])
    drop_rows1=[]
    drop_rows2=[]
    for idx, row in data_df.iterrows():
        steering = getattr(row, 'steering')
        throttle = getattr(row, 'throttle')
        speed=getattr(row, "speed")
        # record the recent steering history
        steering_history.append(steering)
        if len(steering_history) > num:
            steering_history.popleft()

        # if just driving in a straight line continue
        # sterring shall be close to zero when straight line
        if len([1 for i in steering_history if abs(i)<0.01])==num:
            drop_rows1.append(idx)
        if throttle<0.95 or speed<28:
            drop_rows2.append(idx)
    drop_rows=drop_rows1+drop_rows2
    train_ds = data_df.drop(data_df.index[drop_rows])            
    test1_ds = data_df.iloc[drop_rows1,:]
    test2_ds = data_df.iloc[drop_rows2,:]
    return train_ds, test1_ds, test2_ds


# In[4]:

data_df2, data_df3, data_df4 =select_data_set(data_df, 5)
data_df2['steering'].plot.hist(bins=50)
data_df2.to_csv("new_driving_log.csv", index=False)
data_df2['steering'].describe()


# In[5]:

data_df3['steering'].plot.hist(bins=50)
data_df3.to_csv("test1_driving_log.csv", index=False)
data_df3['steering'].describe()


# In[6]:

data_df4['steering'].plot.hist(bins=10)
data_df4.to_csv("test2_driving_log.csv", index=False)
data_df4['steering'].describe()


# In[7]:

samples = []
with open('./new_driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        samples.append(line)

test1_samples = []
with open('./test1_driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        test1_samples.append(line)

test2_samples = []
with open('./test2_driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        test2_samples.append(line)
        
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
validation2_samples=validation_samples+test1_samples+test2_samples
print("Train sampes:",len(train_samples))
print("Valid sampes:",len(validation_samples))
print("Test1 sampes[straight]:",len(test1_samples))
print("Test2 sampes[slow in curve]:",len(test2_samples))


# In[8]:


def crop_img(img, crop_height=66, crop_width=300):
    height = img.shape[0]
    width = img.shape[1]
    y_start=60
    x_start=int(width/2)-int(crop_width/2)
    img=img[y_start:y_start+crop_height,x_start:x_start+crop_width]
    img=cv2.resize(img, (200,66))
    return img 

def gen_new_data_set(mypath, csv_file):
    X_data=[]
    y_data=[]
    f = []
    for (dirpath, dirnames, filenames) in walk(mypath):
        f.extend(filenames)
        break
    #print (f)
    t_list={}
    with open(csv_file,"r") as fi:
        reader=csv.reader(fi)
        for row in reader:
            if(row[1]!='center'):
                fn=re.sub(r'.*center', 'center',row[0])
                t_list[fn]=row[3]

    clip_data=[]
    ke=list(t_list.keys())
    ke.sort()
    for i in ke:
        #print (i)
        if(re.search(r'center', i)):
            fn=mypath+'/'+i
            steering=float(t_list[i])
            img=cv2.imread(fn)
            img=crop_img(img)
            img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X_data.append(img)
            y_data.append(steering)
            if abs(steering) > 0.1:
                image = cv2.flip(img, 1)
                steering = -steering
                X_data.append(img)
                y_data.append(steering)
    X_data=np.array(X_data)
    y_data=np.array(y_data)
    return X_data, y_data       
    
def test_new_data_set(mypath):
    f = []
    for (dirpath, dirnames, filenames) in walk(mypath):
        f.extend(filenames)
        break
    #print (f)
    t_list={}
    #new.csv add index column as first column
    with open("new_driving_log.csv","r") as fi:
        reader=csv.reader(fi)
        for row in reader:
            if(row[1]!='center'):
                fn=re.sub(r'.*center', 'center',row[0])
                t_list[fn]=row[3]
    
    out_folder='./Center_Img'

    clip_data=[]
    ke=list(t_list.keys())
    ke.sort()
    for i in ke:
        #print (i)
        if(re.search(r'center', i)):
            fn=mypath+'/'+i
            #print (fn)
            img=cv2.imread(fn)
            img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img=crop_img(img)
            #cv2.imwrite(out_folder+'/'+i,img)
            txt=str(t_list[i])
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            cv2.putText(img,txt,(0,0), font, 6,(255,255,255),2,cv2.LINE_AA)
            clip_data.append(img)
    clip = ImageSequenceClip(clip_data, fps=50)
    get_ipython().magic("time clip.write_videofile(out_folder+'.mp4', audio=False)")


# In[9]:

a=np.random.randint(1,5, 16)
for i in range(16):
    dx=np.random.randint(2)
    if(dx==1):
        a[i]=-a[i]
print(a)


# In[10]:

ang_range=10
def rotate_img(img,steering,dir_val):
    (rows,cols,c)=img.shape
    if(dir_val=='center'):
        ang_rot = np.random.uniform(ang_range)-ang_range/2
    elif(dir_val=='left'):
        ang_rot = np.random.uniform(ang_range)
    else:
        ang_rot = -np.random.uniform(ang_range)
    # positive: anti-clockwise
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows),ang_rot,1)
    img1 = cv2.warpAffine(img,Rot_M,(cols,rows))
    steering1=steering-int(ang_rot)*0.02
    return img1, steering1
    
def jitter_img_left(img,steering):            
    dx=np.random.randint(0,8)
    #dx=0
    dx1=-4*dx
    steering1=steering+dx1*0.01
    (rows,cols,c)=img.shape
    M = np.float32([[1,0,dx1],[0,1,0]])
    img1 = cv2.warpAffine(img,M,(cols,rows))
    #img1, steering1 = rotate_img(img1, steering1, 'left')
    return img1, steering1

def jitter_img_right(img,steering):            
    dx=np.random.randint(0,8)
    #dx=0
    dx1=4*dx
    steering1=steering+dx1*0.01
    (rows,cols,c)=img.shape
    M = np.float32([[1,0,dx1],[0,1,0]])
    img1 = cv2.warpAffine(img,M,(cols,rows))
    #img1, steering1 = rotate_img(img1, steering1, 'right')
    
    return img1, steering1


def jitter_img(img,steering):            
    dx=np.random.randint(0,64)-32
    steering1=steering+dx*0.01
    (rows,cols,c)=img.shape
    M = np.float32([[1,0,dx],[0,1,0]])
    img1 = cv2.warpAffine(img,M,(cols,rows))
    return img, steering, img1, steering1

def dataset_gen(samples, batch_size=2):
    num_samples = len(samples)
    np.random.seed(1234)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        # double the data set, by flipping images
        for offset in range(0, num_samples, int(batch_size/2)):
            batch_samples = samples[offset:offset+int(batch_size/2)]
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data/data/IMG/'+batch_sample[0].split('/')[-1]
                center_angle = float(batch_sample[3])
                #print(name)
                img = cv2.imread(name)
                img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img=crop_img(img)
                images.append(img)
                angles.append(center_angle)

                name = './data/data/IMG/'+batch_sample[1].split('/')[-1]
                center_angle = float(batch_sample[3])+0.1
                img = cv2.imread(name)
                img=crop_img(img)
                img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                angles.append(center_angle)

                name = './data/data/IMG/'+batch_sample[2].split('/')[-1]                    
                center_angle = float(batch_sample[3])-0.1
                img = cv2.imread(name)
                img=crop_img(img)
                img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)        
                angles.append(center_angle)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield X_train, y_train

def read_ori(camera, batch_sample):                
    if(camera=='center'):
        name = './data/data/IMG/'+batch_sample[0].split('/')[-1]
        center_angle = float(batch_sample[3])
    elif(camera=='left'):
        name = './data/data/IMG/'+batch_sample[1].split('/')[-1]
        center_angle = float(batch_sample[3])+0.2
    else: # right
        name = './data/data/IMG/'+batch_sample[2].split('/')[-1]                    
        center_angle = float(batch_sample[3])-0.2
    #print(name)
    img = cv2.imread(name)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, center_angle

def generator(samples, camera='center', batch_size=128):
    num_samples = len(samples)
    np.random.seed(1234)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        # double the data set, by flipping images
        for offset in range(0, num_samples, int(batch_size/2)):
            batch_samples = samples[offset:offset+int(batch_size/2)]
            images = []
            angles = []
            for batch_sample in batch_samples:
                img,center_angle=read_ori(camera, batch_sample)
                if(abs(center_angle)>0.1):
                    rnd=np.random.randint(2)
                    if(rnd==0):
                        img1=img
                        center_angle1=center_angle
                    else:
                        # right turn
                        if(center_angle>0):
                            img1,center_angle1=jitter_img_right(img,center_angle)
                        else:
                            img1,center_angle1=jitter_img_left(img,center_angle)
                    img1=crop_img(img1)
                    images.append(img1)
                    angles.append(center_angle1)
                    images.append(cv2.flip(img1, 1))
                    angles.append(-1.0*center_angle1)
                else:
                    rnd=np.random.randint(3)
                    if(rnd==0 or rnd==1):
                        img1,center_angle1,img2,center_angle2=jitter_img(img,center_angle)
                        img1 = crop_img(img1)
                        images.append(img1)        
                        angles.append(center_angle1)
                        img2 = crop_img(img2)
                        images.append(img2)        
                        angles.append(center_angle2)
                    else:
                        img1,center_angle1=read_ori('left', batch_sample)
                        img1 = crop_img(img1)
                        images.append(img1)
                        angles.append(center_angle1)
                        img1,center_angle1=read_ori('right', batch_sample)
                        img1 = crop_img(img1)
                        images.append(img1)
                        angles.append(center_angle1)
                        
            X_train = np.array(images)
            y_train = np.array(angles)
            X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
            yield X_train, y_train
batch_size=128
# compile and train the model using the generator function
train_generator = generator(train_samples, 'center', batch_size=batch_size)
valid_generator = generator(validation_samples, 'center', batch_size=batch_size)
valid2_generator = generator(validation2_samples, 'center', batch_size=batch_size)
test1_generator = generator(test1_samples, 'center', batch_size=batch_size)
test2_generator = generator(test2_samples, 'center', batch_size=batch_size)
test_left_generator = generator(train_samples, 'left', batch_size=batch_size)
test_right_generator = generator(train_samples, 'right', batch_size=batch_size)


# In[11]:

batch_size=2
# compile and train the model using the generator function
train_generator1 = generator(train_samples, 'center', batch_size=batch_size)

def test_gen():
    frame_no=0
    show_frame_no=10
    fig_no=2
    #fig_no=3
    #for i in dataset_gen(train_samples):
    for i in train_generator1:
        (img,ster)=i
        plt.figure(figsize=(8,6))
        for k in range(fig_no):
            plt.subplot(1,fig_no,k+1)
            plt.title(str(ster[k]))
            plt.imshow(img[k])
        plt.show()
        if(frame_no==show_frame_no):
            break
        frame_no+=1
test_gen()        


# In[42]:

def pred_generator(samples, camera='center', batch_size=128):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                if(camera=='center'):
                    name = './data/data/IMG/'+batch_sample[0].split('/')[-1]
                    center_angle = float(batch_sample[3])
                elif(camera=='left'):
                    name = './data/data/IMG/'+batch_sample[1].split('/')[-1]
                    center_angle = float(batch_sample[3])+0.2
                else: # right
                    name = './data/data/IMG/'+batch_sample[2].split('/')[-1]                    
                    center_angle = float(batch_sample[3])-0.2

                #print(name)
                img = cv2.imread(name)
                img=crop_img(img)
                img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                angles.append(center_angle)
            X_train = np.array(images)
            yield X_train
            
def actual_steering(samples, camera='center'):
    #left:+0.2; right:-0.2; center: 0
    if(camera=='center'):
        adj_angle=0.0
    elif(camera=='left'):
        adj_angle=0.2
    else:
        adj_angle=-0.2
    angles=[float(x[3])+adj_angle for x in samples]
    y_data = np.array(angles)
    return y_data
            
# compile and train the model using the generator function
pred_train_generator = pred_generator(train_samples, 'center', batch_size=128)
pred_valid_generator = pred_generator(validation_samples, 'center', batch_size=128)
pred_test1_generator = pred_generator(test1_samples, 'center', batch_size=128)
pred_test2_generator = pred_generator(test2_samples, 'center', batch_size=128)
pred_train_left_generator = pred_generator(train_samples, 'left', batch_size=128)
pred_train_right_generator = pred_generator(train_samples, 'right', batch_size=128)


# In[30]:

import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import json
import random

from pathlib import PurePosixPath
from collections import deque

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers import Convolution2D
from keras.optimizers import Adam
#from keras.utils.visualize_util import plot
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

def build_nvidia_model(img_height=66, img_width=200, img_channels=3,
                       dropout=.4, lrate=0.0005):

    # build sequential model
    model = Sequential()

    # normalisation layer
    img_shape = (img_height, img_width, img_channels)
    model.add(Lambda(lambda x: x * 1./127.5 - 1,
                     input_shape=(img_shape),
                     output_shape=(img_shape), name='Normalization'))

    # convolution layers with dropout
    nb_filters = [24, 36, 48, 64, 64]
    kernel_size = [(5, 5), (5, 5), (5, 5), (3, 3), (3, 3)]
    same, valid = ('same', 'valid')
    padding = [valid, valid, valid, valid, valid]
    strides = [(2, 2), (2, 2), (2, 2), (1, 1), (1, 1)]

    for l in range(len(nb_filters)):
        model.add(Convolution2D(nb_filters[l],
                                kernel_size[l][0], kernel_size[l][1],
                                border_mode=padding[l],
                                subsample=strides[l],
                                activation='elu'))
        #model.add(Dropout(dropout))

    # flatten layer
    #1164
    model.add(Flatten())

    # fully connected layers with dropout
    model.add(Dropout(dropout))
    neurons = [100, 50, 10]
    model.add(Dense(neurons[0], activation='elu'))
    model.add(Dense(neurons[1], activation='elu'))
    model.add(Dense(neurons[2], activation='elu'))

    # logit output - steering angle
    model.add(Dense(1,name='Out'))

    optimizer = Adam(lr=lrate)
    model.compile(optimizer=optimizer,
                  loss='mse')
    return model


def get_callbacks():
 
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0,
                                  patience=1, verbose=1, mode='auto')
    return [earlystopping]




# In[33]:

from keras.models import load_model

def train_process(nb_epoch,lrate,dout=0.4,restore=False):
    if(restore==True):
        model = load_model('model.h5')
    else:
        model = build_nvidia_model(dropout=dout,lrate=lrate)
    #for l in model.layers:
    #     print(l.name, l.input_shape, l.output_shape,
    #           l.activation if hasattr(l, 'activation') else 'none')
    history_object = model.fit_generator(
        train_generator,
        samples_per_epoch=2*len(train_samples),
        nb_epoch=nb_epoch,
        callbacks=get_callbacks(),
        validation_data=valid_generator,
        nb_val_samples=2*len(validation_samples))

    print("Fit done!")
    model.save("model.h5")
    print("Model saved!")
    
    return history_object


# In[34]:

#from keras.utils.visualize_util import plot
model = load_model('model.h5')
print(model.summary())
#plot(model, to_file='model.png')


# In[35]:

import time
restore=False
#restore=True
valid_generator=valid2_generator
t1=time.time()
history_object=train_process(5,0.0005,dout=0.6,restore=restore)
t2=time.time()
print("%.2f sec used"%(t2-t1))
#if __name__ == '__main__':
#    tf.app.run()


# In[36]:

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


# In[37]:

from keras.models import load_model

model = load_model('model.h5')
t1=time.time()
metrics = model.evaluate_generator(test1_generator, len(test1_samples))
t2=time.time()
print("Evaluate Straight: %.2f sec used, Error:%.5f"%(t2-t1, metrics))
t1=time.time()
metrics = model.evaluate_generator(train_generator, len(train_samples))
t2=time.time()
print("Evaluate Traing: %.2f sec used, Error:%.5f"%(t2-t1, metrics))
t1=time.time()
metrics = model.evaluate_generator(valid_generator, len(validation_samples))
t2=time.time()
print("Evaluate Validation: %.2f sec used, Error:%.5f"%(t2-t1, metrics))


# In[38]:

def get_integral(li):
    lo=[np.sum(li[:i+1]) for i in range(li.shape[0])]
    return lo


# In[43]:


def plot_steering(steering_pred, steering_actual, t, integral=0):
    if(integral==1):
        lo1=get_integral(steering_pred)
        lo2=get_integral(steering_actual)
    else:
        lo1=steering_pred
        lo2=steering_actual
    plt.plot(lo1)
    plt.plot(lo2)
    plt.title('%s steering'%t)
    plt.ylabel('actual steering')
    plt.xlabel('frame no')
    plt.legend(['predict', 'actual'], loc='upper right')
    plt.show()    

def pred_process(gen, samples, camera, t):
    t1=time.time()
    s=len(samples)
    pred = model.predict_generator(gen, s)
    steering_pred = np.array(pred)
    steering_actual = actual_steering(samples, camera)
    mse=np.mean((steering_pred-steering_actual)**2)
    t2=time.time()
    print("Predict %s Samples %d: %.2f sec used, MSE=%.5f Mean=%.5f, Actual Mean=%.5f"%          (t, steering_pred.shape[0],t2-t1,            mse,           np.mean(steering_pred),            np.mean(steering_actual)))
    plot_steering(steering_pred, steering_actual, t, integral=1)
    return steering_pred, steering_actual



steering_pred_train = pred_process(pred_train_generator, train_samples, 'center', "Train center")
steering_pred_valid = pred_process(pred_valid_generator, validation_samples, 'center', "Valid center")
steering_pred_train_left = pred_process(pred_train_left_generator, train_samples, 'left', "Train Left")
steering_pred_train_right = pred_process(pred_train_right_generator, train_samples, 'right', "Train Right")
steering_pred_test1 = pred_process(pred_test1_generator, test1_samples, 'center', "Straight")
steering_pred_test2 = pred_process(pred_test2_generator, test2_samples, 'center', "Slow curve")



# In[ ]:



