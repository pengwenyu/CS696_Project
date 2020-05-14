from skimage import io,transform
import glob
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import time
import matplotlib.pyplot as plt

#enable or disable gpu for tensflow
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
path='./'
#model output file
model_path='./Alexnet/model.ckpt'

#width,highth,dimension
w=28
h=28
c=3

dropout = 0.75

endpoint = 512

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
        imgs.append(img)
        labels.append(idx)
    idx=1;
    list = os.listdir("./wearGlass")
    for i in range(0, len(list)):
        if i==endpoint:
            break
        imgName = os.path.basename(list[i])
        print('reading the images:%s' % (imgName))
        img = io.imread('./wearGlass/'+imgName)
        img = transform.resize(img, (w, h))
        imgs.append(img)
        labels.append(idx)
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

n_classes = 2

# define convolution
def conv2d(name,x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    # use relu
    return tf.nn.relu(x,name=name)

# MaxPooling
def maxpool2d(name,x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME',name=name)

# normalize
def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0,beta=0.75, name=name)


weights = {
    'wc1': tf.Variable(tf.random_normal([11, 11, 1, 96])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384])),
    'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384])),
    'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256])),
    'wd1': tf.Variable(tf.random_normal([4*4*256, 4096])),
    'wd2': tf.Variable(tf.random_normal([4096, 1024])),
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([96])),
    'bc2': tf.Variable(tf.random_normal([256])),
    'bc3': tf.Variable(tf.random_normal([384])),
    'bc4': tf.Variable(tf.random_normal([384])),
    'bc5': tf.Variable(tf.random_normal([256])),
    'bd1': tf.Variable(tf.random_normal([4096])),
    'bd2': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def alex_net(x, weights, biases, dropout):

    conv1 = conv2d('conv1', x, weights['wc1'], biases['bc1'])

    pool1 = maxpool2d('pool1', conv1, k=2)

    norm1 = norm('norm1', pool1, lsize=4)

    conv2 = conv2d('conv2', norm1, weights['wc2'], biases['bc2'])

    pool2 = maxpool2d('pool2', conv2, k=2)

    norm2 = norm('norm2', pool2, lsize=4)

    conv3 = conv2d('conv3', norm2, weights['wc3'], biases['bc3'])

    norm3 = norm('norm3', conv3, lsize=4)

    conv4 = conv2d('conv4', norm3, weights['wc4'], biases['bc4'])

    conv5 = conv2d('conv5', conv4, weights['wc5'], biases['bc5'])

    pool5 = maxpool2d('pool5', conv5, k=2)

    norm5 = norm('norm5', pool5, lsize=4)



    fc1 = tf.reshape(norm5, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 =tf.add(tf.matmul(fc1, weights['wd1']),biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # dropout
    fc1=tf.nn.dropout(fc1,dropout)


    fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[0]])
    fc2 =tf.add(tf.matmul(fc2, weights['wd2']),biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    # dropout
    fc2=tf.nn.dropout(fc2,dropout)


    out = tf.add(tf.matmul(fc2, weights['out']) ,biases['out'])
    return out


x=tf.placeholder(tf.float32,shape=[None,w,h,c],name='x')
y_=tf.placeholder(tf.int32,shape=[None,],name='y_')
#dropout (keep probability)
keep_prob = tf.placeholder(tf.float32,name='keep_prob')


logits = alex_net(x, weights, biases, keep_prob)

#define name for use
b = tf.constant(value=1,dtype=tf.float32)
logits_eval = tf.multiply(logits,b,name='logits_eval')

loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_)
train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1),tf.int32), y_)
acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#define times to pick up dataset
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

# training and validation
total_time = 0;
n_epoch=10
batch_size=64
saver=tf.train.Saver()
sess=tf.Session()
sess.run(tf.global_variables_initializer())

train_loss_arr = []
train_acc_arr = []
val_loss_arr = []
val_acc_arr = []

for epoch in range(n_epoch):
    start_time = time.time()

    print("====epoch %d====="%epoch)

    #training
    train_loss, train_acc, n_batch = 0, 0, 0
    for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
        _,err,ac=sess.run([train_op,loss,acc], feed_dict={x: x_train_a, y_: y_train_a,keep_prob:dropout})
        train_loss += err; train_acc += ac; n_batch += 1
    print("   train loss: %f" % (np.sum(train_loss)/ n_batch))
    print("   train acc: %f" % (np.sum(train_acc)/ n_batch))
    train_loss_arr.append( np.sum(train_loss)/ n_batch)
    train_acc_arr.append(np.sum(train_acc)/ n_batch)
    #validation
    val_loss, val_acc, n_batch = 0, 0, 0
    for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
        err, ac = sess.run([loss,acc], feed_dict={x: x_val_a, y_: y_val_a,keep_prob: 1})
        val_loss += err; val_acc += ac; n_batch += 1
    print("   validation loss: %f" % (np.sum(val_loss)/ n_batch))
    print("   validation acc: %f" % (np.sum(val_acc)/ n_batch))
    val_loss_arr.append(np.sum(val_loss)/ n_batch)
    val_acc_arr.append(np.sum(val_acc)/ n_batch)
    end_time = time.time()
    print('for one epoch the time cost is', end_time - start_time, 's')
    total_time += end_time - start_time
saver.save(sess,model_path)
sess.close()
print('the total time is',total_time)

# # evaluate
plt.plot(train_acc_arr)
plt.plot(val_acc_arr)
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# 绘制训练 & 验证的损失值
plt.plot(train_loss_arr)
plt.plot(val_loss_arr)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
