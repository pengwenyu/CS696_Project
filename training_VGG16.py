from tensorflow.keras.applications.vgg16 import VGG16
from skimage import io,transform
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from tensorflow.keras.preprocessing import image
import random
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
#enable or disable gpu for tensflow
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
path='./'
#model output file
model_path='./ResNet/model.ckpt'

#width,highth,dimension
w=100
h=100
c=3

endpoint = 5000

#read image and put label
def read_img():
    imgs=[]
    labels=[]
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
        labels.append((0,1))
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
        labels.append((1,0))
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

# using model ResNet50
model = VGG16(
    weights=None,
    classes=2,
    input_shape=(100, 100, 3),
)

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
# # train

history= model.fit(x_train, y_train,validation_data=(x_val,y_val), epochs=10, batch_size=16)

fig = plt.figure()
# plot acc
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# plot loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

model.save('./VGG/my_vgg_model.h5')

img_path="./testing_wearGlass/154613.jpg"

img = image.load_img(img_path, target_size=(100, 100))

img = image.img_to_array(img)/ 255.0
img = np.expand_dims(img, axis=0)

print(model.predict(img))
np.argmax(model.predict(img))

