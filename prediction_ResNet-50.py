from tensorflow.keras.applications.inception_v3 import InceptionV3
from skimage import io,transform
import glob
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time
from tensorflow.keras.preprocessing import image
import random
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)

w=224
h=224
c=3


endpoint= 3000


def read_no_glass_img():
    imgs = []
    list = os.listdir("./testing_noGlass")
    for i in range(0, len(list)):
        if i==endpoint:
            break
        imgName = os.path.basename(list[i])
        print('reading the images:%s' % (imgName))
        img = io.imread('./testing_noGlass/'+imgName)
        img = transform.resize(img, (w, h))
        img = image.img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        imgs.append(img)
    return np.asarray(imgs, np.float32)

def read_wear_glass_img():
    imgs = []
    list = os.listdir("./testing_wearGlass")
    for i in range(0, len(list)):
        if i==endpoint:
            break
        imgName = os.path.basename(list[i])
        print('reading the images:%s' % (imgName))
        img = io.imread('./testing_wearGlass/'+imgName)
        img = transform.resize(img, (w, h))
        img = image.img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        imgs.append(img)
    return np.asarray(imgs, np.float32)


model = tf.keras.models.load_model('./ResNet/my_resnet_model.h5')

data = []
data = read_wear_glass_img()

groundtruth= 0;
output = []

matrix1 =[0]
matrix = [0]
for i in range(0,len(data)):
    matrix1 = model.predict(data[i])
    matrix = matrix1[0]
    if matrix[0]> matrix[1]:
        groundtruth= groundtruth+1
print('the number of groundtruth is ',groundtruth)
print('the number of wrong truth is ',endpoint-groundtruth)
print('accurancy is', groundtruth / endpoint)
print('accurancy is', 1- (groundtruth / endpoint))