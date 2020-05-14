from skimage import io,transform
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import shutil

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"


face_dict = {1:'Has Glass',0:'No Glass'}

w=28
h=28
c=3

endpoint= 3000

def read_no_glass_img():
    imgs = []
    list = os.listdir("./testing_noGlass")
    for i in range(0, len(list)):
        if i ==endpoint:
            break
        imgName = os.path.basename(list[i])
        print('reading the images:%s' % (imgName))
        img = io.imread('./testing_noGlass/'+imgName)
        img = transform.resize(img, (w, h))
        imgs.append(img)
    return np.asarray(imgs, np.float32)

def copy_wrong_noglass_img(index):
    list = os.listdir("./testing_noGlass")
    for i in range(0, len(list)):
        if(i==index):
            imgName = os.path.basename(list[i])
            print('wrong image is:%s' % (imgName))
            oldname= "./testing_noGlass/"+imgName
            newname="./wrongnoGlass/"+imgName
            shutil.copy(oldname, newname)

def read_wear_glass_img():
    imgs = []
    list = os.listdir("./testing_wearGlass")
    for i in range(0, len(list)):
        if i ==endpoint:
            break
        imgName = os.path.basename(list[i])
        print('reading the images:%s' % (imgName))
        img = io.imread('./testing_wearGlass/'+imgName)
        img = transform.resize(img, (w, h))
        imgs.append(img)
    return np.asarray(imgs, np.float32)

def copy_wrong_glass_img(index):
    list = os.listdir("./testing_wearGlass")
    for i in range(0, len(list)):
        if(i==index):
            imgName = os.path.basename(list[i])
            print('wrong image is:%s' % (imgName))
            oldname= "./testing_wearGlass/"+imgName
            newname="./wrongGlass/"+imgName
            shutil.copy(oldname, newname)

with tf.Session() as sess:
    data = []
    data = read_wear_glass_img()
    #data = read_wear_glass_img()
    saver = tf.train.import_meta_graph('./Alexnet/model.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./Alexnet/'))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    y_conv = graph.get_tensor_by_name('y_:0')
    logits = graph.get_tensor_by_name("logits_eval:0")
    feed_dict = {x:data,keep_prob:1}

    classification_result = sess.run(logits,feed_dict)

    #print prediction matrix
    print(classification_result)
    #print max index
    print(tf.argmax(classification_result,1).eval())
    #figure out class
    groundtruth= 0;
    output = []
    output = tf.argmax(classification_result,1).eval()
    for i in range(len(output)):
        if output[i]==0:
            groundtruth=groundtruth+1
            #copy_wrong_noglass_img(i)
    print(groundtruth)
    print(endpoint-groundtruth)
    print('accurancy is', groundtruth / endpoint)
    #print('accurancy is',1- groundtruth/len(output))


    print('system done')