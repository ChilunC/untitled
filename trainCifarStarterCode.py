from scipy import misc
import numpy as np
import os
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import matplotlib as mp
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# --------------------------------------------------
# setup

def variable_summaries(var):
    #"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def weight_variable(shape):
    '''
    Initialize weights
    :param shape: shape of weights, e.g. [w, h ,Cin, Cout] where
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters
    Cout: the number of filters
    :return: a tensor variable for weights with initial values
    '''

    # IMPLEMENT YOUR WEIGHT_VARIABLE HERE
    initial = tf.truncated_normal(shape, stddev=0.1)
    W = tf.Variable(initial)
    return W

def bias_variable(shape):
    '''
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    '''

    # IMPLEMENT YOUR BIAS_VARIABLE HERE
    initial = tf.constant(0.1, shape=shape)
    b = tf.Variable(initial)
    return b

def conv2d(x, W):
    '''
    Perform 2-D convolution
    :param x: input tensor of size [N, W, H, Cin] where
    N: the number of images
    W: width of images
    H: height of images
    Cin: the number of channels of images
    :param W: weight tensor [w, h, Cin, Cout]
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters = the number of channels of images
    Cout: the number of filters
    :return: a tensor of features extracted by the filters, a.k.a. the results after convolution
    '''

    # IMPLEMENT YOUR CONV2D HERE
    h_conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    return h_conv

def max_pool_2x2(x):
    '''
    Perform non-overlapping 2-D maxpooling on 2x2 regions in the input data
    :param x: input data
    :return: the results of maxpooling (max-marginalized + downsampling)
    '''

    # IMPLEMENT YOUR MAX_POOL_2X2 HERE
    h_max = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return h_max


ntrain = 1000 # per class
ntest =  100# per class
nclass = 2 # number of classes
imsize = 28
nchannels = 1
batchsize = 1000
result_dir = './results/' # directory where the results from the training are saved

Train = np.zeros((ntrain*nclass,imsize,imsize,nchannels))
Test = np.zeros((ntest*nclass,imsize,imsize,nchannels))
LTrain = np.zeros((ntrain*nclass,nclass))
LTest = np.zeros((ntest*nclass,nclass))

itrain = -1
itest = -1
for iclass in range(0, nclass):
    for isample in range(0, ntrain):
        #print(os.path.dirname(os.path.realpath("trainCifarStarterCode.py")))
        #dir = os.path.dirname(os.path.realpath("trainCifarStarterCode.py"))
        #dir.replace('\\',"/")
        #print(dir)
        path = 'CIFAR10/CIFAR10/Train/%d/Image%05d.png' % (iclass,isample)
        #filename = os.path.join(dir, 'Image00000.png')
        #print(filename)
        im = misc.imread(path); # 28 by 28
        im = im.astype(float)/255
        itrain += 1
        Train[itrain,:,:,0] = im
        LTrain[itrain,iclass] = 1 # 1-hot lable
    for isample in range(0, ntest):
        path = 'CIFAR10/CIFAR10/Test/%d/Image%05d.png' % (iclass,isample)
        im = misc.imread(path); # 28 by 28
        im = im.astype(float)/255
        itest += 1
        Test[itest,:,:,0] = im
        LTest[itest,iclass] = 1 # 1-hot lable

sess = tf.InteractiveSession()

tf_data = tf.placeholder(tf.float32, shape=[None, imsize, imsize, nchannels]) #tf.reshape(tf.cast(Train, tf.float32), [-1, imsize, imsize, nchannels])#tf variable for the data, remember shape is [None, width, height, numberOfChannels]
tf_labels = tf.placeholder(tf.float32, shape=[None, nclass]) #LTest#tf variable for labels

# --------------------------------------------------
# model
#create your model
# reshape the input image
#x_image = tf.reshape(x, [-1, 28, 28, 1])

# first convolutional layer
W_conv1 = weight_variable([5, 5, nchannels, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(tf_data, W_conv1) + b_conv1)#tf.nn.tanh(conv2d(x_image, W_conv1)+ b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

 # Adding a name scope ensures logical grouping of the layers in the graph.
with tf.name_scope('ConvLayer1'):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
        #weights = weight_variable([input_dim, output_dim])
        variable_summaries(W_conv1)
    with tf.name_scope('biases'):
        #biases = bias_variable([output_dim])
        variable_summaries(b_conv1)
    with tf.name_scope('Wx_plus_b'):
        variable_summaries(h_conv1)
    with tf.name_scope('max_pool_2x2'):
        variable_summaries(h_pool1)

# second convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) #tf.nn.tanh(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# densely connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# densely connected layer
W_fc2 = weight_variable([1024, nclass])
b_fc2 = bias_variable([nclass])
h_pool3_flat = tf.reshape(h_fc1, [-1, 1024])
h_fc2 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc2) + b_fc2)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# softmax maybe?
W_fc3 = weight_variable([nclass, nclass])  # [w, h ,Cin, Cout]
b_fc3 = bias_variable([nclass])
y_conv = tf.matmul(h_fc2, W_fc3) + b_fc3


# --------------------------------------------------
# loss
#set up the loss, optimization, evaluation, and accuracy
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_labels, logits=y_conv))
#tf.summary.scalar(cross_entropy.op.name, cross_entropy)
tf.summary.scalar("cross_entropy", cross_entropy)
optimizer = tf.train.AdagradOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(tf_labels,1))

with tf.name_scope('accuracy'):
    with tf.name_scope('cross_prediction'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.name_scope('accuracy'):
        summary_op = tf.summary.merge_all()
tf.summary.scalar('accuracy', accuracy)
#init = tf.global_variables_initializer()

# Create a saver for writing training checkpoints.
saver = tf.train.Saver()

# Instantiate a SummaryWriter to output summaries and the Graph.
#summary_writer = tf.train.SummaryWriter(result_dir, sess.graph)
summary_writer = tf.summary.FileWriter(result_dir, sess.graph)
# --------------------------------------------------
# optimization

sess.run(tf.global_variables_initializer())
batch_xs = np.zeros([batchsize,imsize,imsize, nchannels])#setup as [batchsize, width, height, numberOfChannels] and use np.zeros()
batch_ys = np.zeros([batchsize,nclass])#setup as [batchsize, the how many classes]
max_step = 5500
nsamples = ntrain*nclass #100  #batchsize
for i in range(max_step): # try a small iteration size once it works then continue
    perm = np.arange(nsamples)
    np.random.shuffle(perm)
    #batch = mnist.train.next_batch(50)
    #print("size")
    #print(len(batch[0]))
    for j in range(batchsize):
        #print("perm")
        #print(perm)
        batch_xs[j,:,:,:] = Train[perm[j],:,:,:]
        batch_ys[j,:] = LTrain[perm[j],:]
    if i%10 == 0:
        #calculate train accuracy and print it
        #print("batch_xs")
        #print(batch_xs.shape)
        #print(type(batch_xs))

        #print("batch_ys")
        #print(batch_ys.shape)
        #print(type(batch_ys))

        #print("tf_data")
        #print(tf_data.shape)
        #print(type(tf_data))

        #print("tf_labels")
        #print(tf_labels.shape)
        #print(type(tf_labels))
        train_accuracy = accuracy.eval(feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
        summary_str, hcon1 = sess.run([summary_op, h_conv1], feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 0.5})
        summary_writer.add_summary(summary_str, i)
        summary_str, centropy = sess.run([summary_op, cross_entropy], feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 0.5})
        summary_writer.add_summary(summary_str, i)
        summary_str, acc = sess.run([summary_op, accuracy], feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 0.5})
        summary_writer.add_summary(summary_str, i)
        summary_writer.flush()
    optimizer.run(feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 0.5}) # dropout only during training
# --------------------------------------------------
# test
summary_str,wcon1 = sess.run([summary_op, W_conv1], feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 0.5})
for b in range(32):
    img = mp.pyplot.imshow(wcon1[:,:,0,b])
    fname = 'Image%05d.png' % (b)
    mp.image.imsave(fname, wcon1[:,:,0,b], vmin=None, vmax=None, cmap=None, format=None, origin=None, dpi=100)

    #raw_input()

print("test accuracy %g"%accuracy.eval(feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1.0}))


sess.close()