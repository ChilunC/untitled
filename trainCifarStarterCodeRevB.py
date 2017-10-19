from scipy import misc
import scipy
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import matplotlib as mp
from PIL import Image
import pylab
import os

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

    return tf.Variable(initial)

def bias_variable(shape):
    '''
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    '''

    # IMPLEMENT YOUR BIAS_VARIABLE HERE
    initial = tf.constant(0.1, shape=shape)

    return tf.Variable(initial)

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


# Loading CIFAR10 images from director

ntrain = 1000
ntest = 100
nclass = 10
imsize = 28
nchannels = 1
batchsize = 50
result_dir = './results/' # directory where the results from the training are saved

Train = np.zeros((ntrain*nclass,imsize,imsize,nchannels))
Test = np.zeros((ntest*nclass,imsize,imsize,nchannels))
LTrain = np.zeros((ntrain*nclass,nclass))
LTest = np.zeros((ntest*nclass,nclass))

itrain = -1
itest = -1
for iclass in range(0, nclass):
    for isample in range(0, ntrain):
        path = 'Data/CIFAR10/Train/%d/Image%05d.png' % (iclass,isample)
        im = misc.imread(path); # 28 by 28
        im = im.astype(float)/255
        itrain += 1
        Train[itrain,:,:,0] = im
        LTrain[itrain,iclass] = 1 # 1-hot lable
    for isample in range(0, ntest):
        path = 'Data/CIFAR10/Test/%d/Image%05d.png' % (iclass,isample)
        im = misc.imread(path); # 28 by 28
        im = im.astype(float)/255
        itest += 1
        Test[itest,:,:,0] = im
        LTest[itest,iclass] = 1 # 1-hot lable

# Show image as a check
#img = Train[2,:,:,0]
#print(LTrain[2,:])
#plt.imshow(img, cmap='gray')
#plt.show()

sess = tf.InteractiveSession()

tf_data = tf.placeholder(tf.float32, shape=[None, imsize, imsize, nchannels])  #tf variable for the data, remember shape is [None, width, height, numberOfChannels]
tf_labels = tf.placeholder(tf.float32, shape=[None, nclass])  #tf variable for labels

# --------------------------------------------------
# model
#create your model

#reshape input image to a 4D tensor
x_image = tf.reshape(tf_data, [-1, imsize, imsize, nchannels])

# First Layer

# Conv layer with kernel 5x5 and 32 filter maps followed by ReLu activation
W_conv1 = weight_variable([5, 5, nchannels, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# Max Pooling layer subsampling by 2
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

# Second Layer

# Conv layer with kernal 5x5 and 64 filter maps followed by ReLu activation
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# FC layer that has input 7*7*64 vector and output 1024
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
# flatten h_pool2 back to a 1xn vector
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Second FC layer that has 1024 input and output 10 classes
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2


# --------------------------------------------------
# loss
with tf.name_scope('cross_prediction'):
    diff = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_labels, logits=y_conv))
    cross_entropy = tf.reduce_mean(diff)
tf.summary.scalar("cross_entropy", cross_entropy)
#set up the loss, optimization, evaluation, and accuracy

optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.arg_max(tf_labels,1))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)
summary_op = tf.summary.merge_all()

init = tf.global_variables_initializer()

# Create a saver for writing training checkpoints.
saver = tf.train.Saver()

summary_writer = tf.summary.FileWriter(result_dir, sess.graph)
# --------------------------------------------------
# optimization

sess.run(init)

batch_xs = np.zeros([batchsize, imsize, imsize, nchannels])  #setup as [batchsize, width, height, numberOfChannels] and use np.zeros()
batch_ys = np.zeros([batchsize, nclass])  #setup as [batchsize, the how many classes]
nsamples = ntrain*nclass
hconTest = []
centropyTest = []
accTest = []

hcon1Test = []
hcon2Test = []
accTest2 = []
mean = []
stddev = []
max = []
min = []
mean2 = []
stddev2 = []
max2 = []
min2 = []
testAcc = []
trainAcc = []
stepAcc = []

max_step = 100
for i in range(max_step):  # try a small iteration size once it works then continue
    perm = np.arange(nsamples)
    np.random.shuffle(perm)
    for j in range(batchsize):
        batch_xs[j,:,:,:] = Train[perm[j],:,:,:]
        batch_ys[j,:] = LTrain[perm[j],:]
    if i%10 == 0:
        #calculate train accuracy and print it
        #print(cross_entropy.op.name)
        #summary_str, hcon1, centropy, acc, wcon1 = sess.run([summary_op, h_conv1, cross_entropy, accuracy, W_conv1], feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 0.5})
        summary_str, hcon1, centropy = sess.run([summary_op, h_conv1, cross_entropy],feed_dict={tf_data: batch_xs, tf_labels: batch_ys,keep_prob: 0.5})
        #summary_str = sess.run(summary_op, feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 0.5})
        #print(acc)
        #hconTest.append(hcon1)
        #centropyTest.append(centropy)
        #accTest.append(acc)

        #summary_str, testacc, hcon1, hcon2 = sess.run([summary_op, accuracy, h_conv1, h_conv2], feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1.0})
        summary_writer.add_summary(summary_str, i)
        summary_writer.flush()
        #summary_str, centropy = sess.run([summary_op, cross_entropy.op.name], feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 0.5})
        #hcon1Test.append(hcon1)
        #hcon2Test.append(hcon2)
        #accTest2.append(testacc)
        #mean.append(tf.reduce_mean(hcon1))
        # tf.summary.scalar('mean', mean)
        #stddev.append(tf.sqrt(tf.reduce_mean(tf.square(hcon1 - mean[i]))))
        #max.append(tf.reduce_max(hcon1))
        #min.append(tf.reduce_min(hcon1))

        #mean2.append(tf.reduce_mean(hcon2))
        # tf.summary.scalar('mean', mean)
        #stddev2.append(tf.sqrt(tf.reduce_mean(tf.square(hcon2 - mean2[i]))))
        #max2.append(tf.reduce_max(hcon2))
        #min2.append(tf.reduce_min(hcon2))

        #print("step %d, training accuracy %g" % (i, acc))
        training_accuracy = accuracy.eval(feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 1.0})
        test_accuracy = accuracy.eval(feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1.0})
        testAcc.append(test_accuracy)
        trainAcc.append(training_accuracy)
        stepAcc.append(i)
        print("Train accuracy %g, after %d:" % (training_accuracy, i))
        #print("test accuracy %g" % (test_accuracy))
        checkpoint_file = os.path.join(result_dir, 'checkpoint')
        saver.save(sess, checkpoint_file, global_step=i)
    optimizer.run(feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 0.5}) # dropout only during training

# --------------------------------------------------
# test
#print(type(trainAcc))
#print(trainAcc)
pl1 = plt.scatter(stepAcc, trainAcc)
fname = 'Train_Accuracy.png'
pylab.savefig(fname)
plt.clf()
pl1 = plt.scatter(stepAcc, testAcc)
fname = 'Test_Accuracy.png'
pylab.savefig(fname)
plt.clf()
#plt.show()

#mp.image.imsave(fname, pl1, vmin=None, vmax=None, cmap=None, format=None, origin=None, dpi=100)
#pl1 = mp.pyplot.plot(centropyTest)
#fname = 'PyPlotcentropy.png'
#mp.pyplot.savefig(fname, bbox_inches='tight')
#mp.image.imsave(fname, pl1, vmin=None, vmax=None, cmap=None, format=None, origin=None, dpi=100)
#pl1 = mp.pyplot.plot(accTest)
#fname = 'PyPlotacc.png'
#mp.pyplot.savefig(fname, bbox_inches='tight')
#mp.image.imsave(fname, pl1, vmin=None, vmax=None, cmap=None, format=None, origin=None, dpi=100)

# test
wcon1 = sess.run([W_conv1], feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 1.0})

for b in range(32):
    im3 = wcon1[0][:,:,0,b]
    #print(im3)
    #print(type(im3))#wcon1[0][:,:,0,b])
    #print(im3.shape)
    #print(type(img))
    fname = 'Weights%05d.png' % (b)
    misc.imsave(fname, im3)
    #print(b)
    #print(len(wcon1[0][0][0][0]))
    #wcon1blah = wcon1[0][:, :, 0, b]
    #print(wcon1blah)
    #img = mp.pyplot.imshow(wcon1[0][:,:,0,b])
    #im2 = Image.new(im.mode, im.size)
    #im2.putdata(img)
    #img.show()
    #img = plt.imshow(im.reshape(wcon1blah.shape[0], wcon1blah.shape[1]), cmap=plt.cm.Greys)
    #img = Image.fromarray(W_conv1[0][:,:,0,b])
    #img = tf.reshape(wcon1[0][:,:,0,b], [5,5])

    #mp.image.imsave(fname, im2, vmin=None, vmax=None, cmap=None, format=None, origin=None, dpi=100)

    #raw_input()

#for i in range(ntest*nclass): # try a small iteration size once it works then continue
    #testacc, hcon1, hcon2 = sess.run([accuracy, h_conv1, h_conv2], feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1.0})
    #hcon1Test.append(hcon1)
    #hcon2Test.append(hcon2)
    #accTest2.append(testacc)
    #mean.append(tf.reduce_mean(hcon1))
    #tf.summary.scalar('mean', mean)
    #stddev.append(tf.sqrt(tf.reduce_mean(tf.square(hcon1 - mean[i]))))
    #max.append(tf.reduce_max(hcon1))
    #min.append(tf.reduce_min(hcon1))

    #mean2.append(tf.reduce_mean(hcon2))
    #tf.summary.scalar('mean', mean)
    #stddev2.append(tf.sqrt(tf.reduce_mean(tf.square(hcon2 - mean2[i]))))
    #max2.append(tf.reduce_max(hcon2))
    #min2.append(tf.reduce_min(hcon2))
    #tf.summary.histogram('histogram', var)
        #summary_writer.add_summary(summary_str, i)
        #summary_str, centropy = sess.run([summary_op, cross_entropy], feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 0.5})
        #summary_writer.add_summary(summary_str, i)
        #summary_str, acc = sess.run([summary_op, accuracy], feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 0.5})
        #summary_writer.add_summary(summary_str, i)
        #summary_writer.flush()
#pl1 = mp.pyplot.plot(hcon1Test)
#fname = 'PyPlothcon1.png'
#mp.image.imsave(fname, pl1, vmin=None, vmax=None, cmap=None, format=None, origin=None, dpi=100)
#print(mean)
#pl1 = mp.pyplot.plot(mean)
#fname = 'PyPlothcon1mean.png'
#mp.image.imsave(fname, pl1, vmin=None, vmax=None, cmap=None, format=None, origin=None, dpi=100)
#pl1 = mp.pyplot.plot(stddev)
#fname = 'PyPlothcon1stddev.png'
#mp.image.imsave(fname, pl1, vmin=None, vmax=None, cmap=None, format=None, origin=None, dpi=100)
#pl1 = mp.pyplot.plot(max)
#fname = 'PyPlothcon1max.png'
#mp.image.imsave(fname, pl1, vmin=None, vmax=None, cmap=None, format=None, origin=None, dpi=100)
#pl1 = mp.pyplot.plot(min)
#fname = 'PyPlothcon1min.png'
#mp.image.imsave(fname, pl1, vmin=None, vmax=None, cmap=None, format=None, origin=None, dpi=100)
#pl1 = mp.pyplot.plot(hcon2Test)
#fname = 'PyPlotcentropy.png'
#mp.image.imsave(fname, pl1, vmin=None, vmax=None, cmap=None, format=None, origin=None, dpi=100)
#pl1 = mp.pyplot.plot(mean2)
#fname = 'PyPlothcon2mean.png'
#mp.image.imsave(fname, pl1, vmin=None, vmax=None, cmap=None, format=None, origin=None, dpi=100)
#pl1 = mp.pyplot.plot(stddev2)
#fname = 'PyPlothcon2stddev.png'
#mp.image.imsave(fname, pl1, vmin=None, vmax=None, cmap=None, format=None, origin=None, dpi=100)
#pl1 = mp.pyplot.plot(max2)
#fname = 'PyPlothcon2max.png'
#mp.image.imsave(fname, pl1, vmin=None, vmax=None, cmap=None, format=None, origin=None, dpi=100)
#pl1 = mp.pyplot.plot(min2)
#fname = 'PyPlothcon2min.png'
#mp.image.imsave(fname, pl1, vmin=None, vmax=None, cmap=None, format=None, origin=None, dpi=100)
#pl1 = mp.pyplot.plot(accTest)
#fname = 'PyPlotacc.png'
#mp.image.imsave(fname, pl1, vmin=None, vmax=None, cmap=None, format=None, origin=None, dpi=100)

print("test accuracy %g" % accuracy.eval(feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1.0}))

sess.close()