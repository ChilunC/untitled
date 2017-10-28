import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import pylab
import matplotlib.pyplot as plt
import matplotlib as mp
import time
import random
import os

from tensorflow.examples.tutorials.mnist import input_data

random.seed(3)
result_dir = './resultsRNN/'  # directory where the results from the training are saved
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)  # call mnist function

#RNN parameters
#learningRate = .02 #.01 .001
#trainingIters = 2000000
#batchSize = 280
#displayStep = 100
#nHidden = 100

#GRU parameters
learningRate = .02 #.01 .001
trainingIters = 2000000
batchSize = 280
displayStep = 100
nHidden = 100
start_time = time.time()
#LSTM parameters
#learningRate = 0.02
#trainingIters = 2000000
#batchSize = 300
#displayStep = 100
#nHidden = 250  # number of neurons for the RNN

nInput = 28  # we want the input to take the 28 pixels
nSteps = 28  # every 28

nClasses = 10  # this is MNIST so you know

x = tf.placeholder('float', [None, nSteps, nInput])
y = tf.placeholder('float', [None, nClasses])
keep_prob = tf.placeholder(tf.float32)

weights = {
    'out': tf.Variable(tf.random_normal([nHidden, nClasses]))
}

biases = {
    'out': tf.Variable(tf.random_normal([nClasses]))
}


def RNN(x, weights, biases):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, nInput])
    #x = tf.split(0, nSteps, x)  # configuring so you can get it as needed for the 28 pixels
    x = tf.split(x, nSteps, 0)

    #lstmCell = tf.contrib.rnn.BasicRNNCell(nHidden)
    lstmCell = tf.contrib.rnn.BasicLSTMCell(nHidden, forget_bias=1.0)  # find which lstm to use in the documentation
    #lstmCell = tf.contrib.rnn.GRUCell(nHidden)
    # words = tf.placeholder(tf.int32, [batch_size, nSteps])
    # state = tf.zeros([batch_size, lstm.state_size])

    outputs, states = tf.contrib.rnn.static_rnn(lstmCell, x, dtype=tf.float32)  # for the rnn where to get the output and hidden state

    # initial_state = state = tf.zeros([batch_size, lstmCell.state_size])

    return tf.matmul(outputs[-1], weights['out']) + biases['out']


pred = RNN(x, weights, biases)

# optimization
# create the cost, optimization, evaluation, and accuracy
# for the cost softmax_cross_entropy_with_logits seems really good
with tf.name_scope('cross_prediction'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
tf.summary.scalar("cost", cost)



batch = tf.Variable(0)

learning_rate = tf.train.exponential_decay(
    0.01,                # Base learning rate.
    batch * batchSize,  # Current index into the dataset.
    nSteps,          # Decay step.
    0.95,                # Decay rate.
    staircase=True)
#optimizer = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(cost,global_step=batch)
#optimizer = tf.train.AdagradOptimizer(1e-4).minimize(cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#optimizer = tf.train.MomentumOptimizer(learningRate, momentum, use_locking=False, name='Momentum', use_nesterov=False)

correctPred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
tf.summary.scalar('accuracy', accuracy)
summary_op = tf.summary.merge_all()



init = tf.global_variables_initializer()

saver = tf.train.Saver()

testAcc = []
trainAcc = []
stepAcc = []
lossTrack = []

with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(result_dir, sess.graph)

    sess.run(init)
    step = 1

    testData = mnist.test.images.reshape((-1, nSteps, nInput))
    testLabel = mnist.test.labels

    while step * batchSize < trainingIters:
        batchX, batchY = mnist.train.next_batch(batchSize)  # mnist has a way to get the next batch
        batchX = batchX.reshape((batchSize, nSteps, nInput))

        #sess.run(optimizer, feed_dict={})

        if step % displayStep == 0:
            accTrain = accuracy.eval(feed_dict={x: batchX, y: batchY, keep_prob: 1.0})
            summary_str,loss = sess.run([summary_op, cost], feed_dict={x: batchX, y: batchY, keep_prob: 0.5})

            summary_writer.add_summary(summary_str, step*batchSize)
            summary_writer.flush()

            summary_writer.flush()
            checkpoint_file = os.path.join(result_dir, 'checkpoint')
            saver.save(sess, checkpoint_file, global_step=step * batchSize)

            print("Iter " + str(step * batchSize) + ", Minibatch Loss= " + \
                  "{:.5f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(accTrain))
            lossTrack.append(loss)
            accTest = accuracy.eval(feed_dict={x: testData, y: testLabel, keep_prob: 1.0})
            testAcc.append(accTest)
            trainAcc.append(accTrain)
            stepAcc.append(step*batchSize)
        optimizer.run(feed_dict={x: batchX, y: batchY, keep_prob: 0.5})  # run one train_step
        step += 1

    print('Optimization finished')


    print("Testing Accuracy:", \
          sess.run(accuracy, feed_dict={x: testData, y: testLabel, keep_prob: 1.0}))

pl1 = plt.scatter(stepAcc, trainAcc)
fname = 'accRNN/Train_Accuracy.png'
pylab.savefig(fname)
plt.clf()
pl1 = plt.scatter(stepAcc, testAcc)
fname = 'accRNN/Test_Accuracy.png'
pylab.savefig(fname)
plt.clf()
pl1 = plt.scatter(stepAcc, lossTrack)
fname = 'accRNN/cross_entropy.png'
pylab.savefig(fname)
plt.clf()

stop_time = time.time()
print('The training takes %f second to finish'%(stop_time - start_time))