import tensorflow as tf
from tensorflow.python.ops import gen_nn_ops
#from tensorflow.python.ops import max_pool_with_argmax_and_mask
import numpy as np

def max_pool(inp, k=2):
    return tf.nn.max_pool_with_argmax(inp, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding="SAME")

def max_unpool(inp, argmax, k=2):
    return unpool2(inp, argmax)
    #return tf.nn.max_unpool(inp, argmax, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding="SAME")
    #return gen_nn_ops._max_unpool(inp, argmax, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding="SAME")


def unravel_argmax(argmax, shape):
    print("argmax")
    print(argmax.get_shape().as_list())

    argmax_shape = argmax.get_shape()
    print(argmax_shape[0])
    #new_1dim_shape = tf.shape(tf.constant(0, shape=[tf.Dimension(4), argmax_shape[0]*argmax_shape[1]*argmax_shape[2]*argmax_shape[3]]))
    new_1dim_shape = tf.shape(tf.constant(0, shape=[tf.Dimension(4), argmax_shape[1] * argmax_shape[2] * argmax_shape[3]]))
    batch_shape = tf.constant(0, dtype=tf.int64, shape=[argmax_shape[0], 1, 1, 1]).get_shape()
    b = tf.multiply(tf.ones_like(argmax), tf.reshape(tf.range(shape[0]), batch_shape))
    y = argmax // (shape[2] * shape[3])
    x = argmax % (shape[2] * shape[3]) // shape[3]
    c = tf.ones_like(argmax) * tf.range(shape[3])
    pack = tf.stack([b, y, x, c])
    pack = tf.reshape(pack, new_1dim_shape)
    pack = tf.transpose(pack)
    return pack

def unpool2(updates, mask, ksize=2, name="unpool"):
    if isinstance(ksize, int):
        ksize = [1, ksize, ksize, 1]
    input_shape = updates.get_shape().as_list()
    #  calculation new shape
    output_shape = [input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3]]
    # calculation indices for batch, height, width and feature maps
    one_like_mask = tf.ones_like(mask)
    bsize = tf.to_int64(tf.shape(updates)[0])
    batch_range = tf.reshape(tf.range(bsize, dtype=tf.int64),
                             shape=[-1, 1, 1, 1])
    b = one_like_mask * batch_range
    y = mask // (output_shape[1] * output_shape[2])
    x = mask % (output_shape[1] * output_shape[2]) // output_shape[2]
    feature_range = tf.range(output_shape[2], dtype=tf.int64)
    f = one_like_mask * feature_range
    # transpose indices & reshape update values to one dimension
    updates_size = tf.size(updates)
    indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
    values = tf.reshape(updates, [updates_size])
    ret = tf.scatter_nd(indices, values, tf.concat(
        [[bsize], tf.to_int64(output_shape)], axis=0))
    return ret

#def unpool(updates, mask, ksize=[1, 2, 2, 1]):
#    input_shape = updates.get_shape()
#    new_dim_y = input_shape[1] * ksize[1]
#    new_dim_x = input_shape[2] * ksize[2]
#    output_shape = tf.to_int64((tf.constant(0, dtype=tf.int64, shape=[input_shape[0], new_dim_y, new_dim_x, input_shape[3]]).get_shape()))
#    indices = unravel_argmax(mask, output_shape)
#    new_1dim_shape = tf.shape(tf.constant(0, shape=[input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]]))
#    values = tf.reshape(updates, new_1dim_shape)
#    ret = tf.scatter_nd(indices, values, output_shape)
#    return ret

#def conv2d(inp, name):
#    w = weights[name]
#    b = biases[name]
#    var = tf.nn.conv2d(inp, w, [1, 1, 1, 1], padding='SAME')
#    var = tf.nn.bias_add(var, b)
#    var = tf.nn.relu(var)
#    return var

def conv2d_transpose(b,w, out_shape):
    #w = weights[name]
    #b = biases[name]

    #dims = inp.get_shape().dims[:3]
    #dims.append(w.get_shape()[-2]) # adpot channels from weights (weight definition for deconv has switched input and output channel!)
    #out_shape = tf.TensorShape(dims)

    #var = tf.nn.conv2d_transpose(inpbias, w, out_shape, strides=[1, 1, 1, 1], padding="SAME")
    return tf.nn.conv2d_transpose(b, w, out_shape, strides=[1, 1, 1, 1], padding="SAME")
    #var = tf.nn.bias_add(var, b)
    #if not dropout_prob is None:
    #    var = tf.nn.relu(var)
    #    var = tf.nn.dropout(var, dropout_prob)
    #return var
def deconvLay1(w,b,out_shape, unPool):
    if unPool == None:
        featuresReLu1 = tf.placeholder("float", [None, 32, 32, 32])
    else:
        featuresReLu1 = unPool
    unReLu = tf.nn.relu(featuresReLu1)
    unBias = unReLu
    unConv = conv2d_transpose(unBias, w, out_shape) #tf.nn.conv2d_transpose(unBias, wConv1, output_shape=[batchsizeFeatures, imagesize, imagesize, colors],strides=[1, 1, 1, 1], padding="SAME")
    return unConv

#not implemented
#def deconvLay1Eval():
    # display features
#    for i in xrange(32):
#        isolated = activations1.copy()
#        isolated[:, :, :, :i] = 0
#        isolated[:, :, :, i + 1:] = 0
#        pixelactive = unConv.eval(feed_dict={featuresReLu1: isolated})
#        totals = np.sum(pixelactive, axis=(1, 2, 3))
#        best = np.argmax(totals, axis=0)
        # best = 0
#        imsave("activ" + str(i) + ".png", pixelactive[best])
#        saveImage(inputImage[best], "activ" + str(i) + "-base.png")

    # display same feature for many images
#    for i in xrange(batchsizeFeatures):
#        isolated = activations1.copy()
#        isolated[:, :, :, :6] = 0
#        isolated[:, :, :, 7:] = 0
#        pixelactive = unConv.eval(feed_dict={featuresReLu1: isolated})
#        totals = np.sum(pixelactive, axis=(1, 2, 3))
#        best = np.argmax(totals, axis=0)
#        imsave("activ" + str(i) + ".png", pixelactive[i])
#        saveImage(inputImage[i], "activ" + str(i) + "-base.png")

def deconvLay2(w,b,out_shape,argmax, featuresReLu2, w2, b2, out_shape2): #, argmax2, argmax_mask2):
    if featuresReLu2 == None:
        featuresReLu2 = tf.placeholder("float", [None, 16, 16, 64])
    unReLu2 = tf.nn.relu(featuresReLu2)
    unBias2 = unReLu2 - b2
    unConv2 = conv2d_transpose(unBias2, w2, out_shape2)
    unPool = max_unpool(unConv2,argmax) #unpool(unConv2)
    #unPool = unpool(unConv2, argmax, ksize=[1, 2, 2, 1])
    #unConv =
    #unConv2 = tf.nn.conv2d_transpose(unBias2, w2,
    #                                 output_shape=[batchsizeFeatures, imagesize / 2, imagesize / 2, 32],
    #                                 strides=[1, 1, 1, 1], padding="SAME")
    return deconvLay1(w,b,out_shape, unPool)
    #unPool = unpool(unConv2)
    #unReLu = tf.nn.relu(unPool)
    #unBias = unReLu
    #unConv = tf.nn.conv2d_transpose(unBias, wConv1, output_shape=[batchsizeFeatures, imagesize, imagesize, colors],
    #                                strides=[1, 1, 1, 1], padding="SAME")
weights = {
    "conv1":    tf.Variable(tf.random_normal([3, 3,  3, 16])),
    "conv2":    tf.Variable(tf.random_normal([3, 3, 16, 32])),
    "conv3":    tf.Variable(tf.random_normal([3, 3, 32, 32])),
    "deconv2":  tf.Variable(tf.random_normal([3, 3, 16, 32])),
    "deconv1":  tf.Variable(tf.random_normal([3, 3,  1, 16])) }

biases = {
    "conv1":    tf.Variable(tf.random_normal([16])),
    "conv2":    tf.Variable(tf.random_normal([32])),
    "conv3":    tf.Variable(tf.random_normal([32])),
    "deconv2":  tf.Variable(tf.random_normal([16])),
    "deconv1":  tf.Variable(tf.random_normal([ 1])) }


## Build Miniature CEDN
x = tf.placeholder(tf.float32, [12, 20, 20, 3])
y = tf.placeholder(tf.float32, [12, 20, 20, 1])
p = tf.placeholder(tf.float32)

#conv1                                   = conv2d(x, "conv1")
#maxp1, maxp1_argmax, maxp1_argmax_mask  = max_pool(conv1)

#conv2                                   = conv2d(maxp1, "conv2")
#maxp2, maxp2_argmax, maxp2_argmax_mask  = max_pool(conv2)

#conv3                                   = conv2d(maxp2, "conv3")
#
#maxup2                                  = max_unpool(conv3, maxp2_argmax, maxp2_argmax_mask)
#deconv2                                 = conv2d_transpose(maxup2, "deconv2", p)

#maxup1                                  = max_unpool(deconv2, maxp1_argmax, maxp1_argmax_mask)
#deconv1                                 = conv2d_transpose(maxup1, "deconv1", None)


## Optimizing Stuff
#loss        = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(deconv1, y))
#optimizer   = tf.train.AdamOptimizer(learning_rate=1).minimize(loss)


## Test Data
np.random.seed(123)
batch_x = np.where(np.random.rand(12, 20, 20, 3) > 0.5, 1.0, -1.0)
batch_y = np.where(np.random.rand(12, 20, 20, 1) > 0.5, 1.0,  0.0)
prob    = 0.5


with tf.Session() as session:
    tf.set_random_seed(123)
    session.run(tf.initialize_all_variables())

    print("\n\n")
    for i in range(10):
        #session.run(optimizer, feed_dict={x: batch_x, y: batch_y, p: prob})
        print("step", i + 1)
        #print("loss",  session.run(loss, feed_dict={x: batch_x, y: batch_y, p: 1.0}), "\n\n")