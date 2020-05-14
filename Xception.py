from tensorflow.keras.applications.xception import Xception
from skimage import io,transform
import glob
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import time
from tensorflow.keras.preprocessing import image
import random
#enable or disable gpu for tensflow
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
path='./'
#model output file
model_path='./Xception/my_Xception_model.h5'

#width,highth,dimension
w=100
h=100
c=3

endpoint=1024
#read image and put label
def read_img():
    imgs=[]
    labels=[]
    idx=0;
    list = os.listdir("./noGlass")
    for i in range(0, len(list)):
        if i==endpoint:
            break
        imgName = os.path.basename(list[i])
        print('reading the images:%s' % (imgName))
        img = io.imread('./noGlass/'+imgName)
        img = transform.resize(img, (w, h))
        img = image.img_to_array(img) / 255.0
        imgs.append(img)
        labels.append((idx,1))
    idx=1;
    list = os.listdir("./wearGlass")
    for i in range(0, len(list)):
        if i==endpoint:
            break
        imgName = os.path.basename(list[i])
        print('reading the images:%s' % (imgName))
        img = io.imread('./wearGlass/'+imgName)
        img = transform.resize(img, (w, h))
        img = image.img_to_array(img) / 255.0
        imgs.append(img)
        labels.append((idx,0))
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)
data,label=read_img()

#random dataset
num_example=data.shape[0]
arr=np.arange(num_example)
np.random.shuffle(arr)
data=data[arr]
label=label[arr]


#80% for training and 20% for validation
ratio=0.8
s=np.int(num_example*ratio)
x_train=data[:s]
y_train=label[:s]
x_val=data[s:]
y_val=label[s:]

index = [i for i in range(len(x_train))]
random.shuffle(index)
x_train = x_train[index]
y_train = y_train[index]

index = [i for i in range(len(x_val))]
random.shuffle(index)
x_val = x_val[index]
y_val = y_val[index]



# using model Xception
model = Xception(
    weights=None,
    classes=2,
    input_shape=(100,100,3),
)

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# # train

history= model.fit(x_train, y_train,validation_data=(x_val,y_val), epochs=10, batch_size=16)

# # evaluate
preds=model.evaluate(x_val,y_val,batch_size=32)
print("loss="+str(preds[0]))
print("accuracy="+str(preds[1]))

model.save(model_path)