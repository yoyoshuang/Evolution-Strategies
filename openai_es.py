# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np

FLAGS = None

# hyperparameters
npop = 50 # population size
sigma = 0.1 # noise standard deviation
alpha = 0.001 # learning rate
scaling = 2**125

np.random.seed(0)

def get_logits(x,params):
  scaling = 2**125
  h1 = tf.nn.bias_add(tf.matmul(x , params[0]), params[1]) / scaling
  h2 = tf.nn.bias_add(tf.matmul(h1, params[2]) , params[3] / scaling)   
  o =   tf.nn.bias_add(tf.matmul(h2, params[4]), params[5]/ scaling)*scaling
  return o

def p2v(par,sess):
  v = []
  for tp in par:
    p = sess.run(tp)
    k = p.flatten()
    v = np.concatenate((v,k),axis=0)
  v = np.reshape(v,(1,len(v)))
  return v[0]

def v2p(v,par,sess):
  start = 0
  ov = []
  # print(v)
  for tp in par:
    p = sess.run(tp)
    if len(p.shape)==2:
      s0,s1 = p.shape    
      end = start+(s0*s1)
    if len(p.shape)==1:
      s0 = p.shape
      end = start+s0[0] 

    vcut = v[start:end]
    vcut = np.reshape(vcut,p.shape)
    ov.append(vcut)
    start = end

    # updatep = tf.assign(tp,vcut)
    # sess.run(updatep) 

  return ov
   
def lossfun(y_,y_conv):
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)
  return cross_entropy

def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  y_ = tf.placeholder(tf.float32, [None, 10])

  w1 = tf.Variable(np.random.normal(scale=np.sqrt(2./784),size=[784,10]).astype(np.float32))
  b1 = tf.Variable(np.zeros(10,dtype=np.float32))

  params = [w1,b1]

  y_conv = tf.nn.bias_add(tf.matmul(x,w1)/scaling,b1/scaling)*scaling

  w1_try = tf.placeholder(tf.float32,[784,10])
  b1_try = tf.placeholder(tf.float32,10)
  reword = tf.nn.bias_add(tf.matmul(x,w1_try),b1_try)/scaling*scaling

  w1new = tf.placeholder(tf.float32,[784,10])
  b1new = tf.placeholder(tf.float32,10)
  uppar1 = tf.assign(w1,w1new)
  uppar2 = tf.assign(b1,b1new)

  # Define loss and optimizer
  with tf.name_scope('loss'):
    cross_entropy = lossfun(y_,y_conv)
    loss_try= lossfun(y_,reword)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  # with tf.Graph().as_default():

  with tf.Session() as sess:
    # global sess
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
      batch = mnist.train.next_batch(50)
      # print("i=========:",i)
      if i % 200== 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1]})
        print('step %d, training accuracy %g' % (i, train_accuracy))

      # es step 1 jitter par using gaussian of sigma 0.1
      vp = p2v(params,sess)
      N = np.random.randn(npop, len(vp)) # samples from a normal distribution N(0,1)
      R = np.zeros(npop)
      for j in range(npop):
        vp_try = np.add(vp,sigma*N[j])
        ov = v2p(vp_try,params,sess)
        R[j] = -1*sess.run(loss_try,feed_dict={x:batch[0], y_: batch[1],
          w1_try:ov[0],b1_try:ov[1]})

      # es step 2 update par by R
      A = (R - np.mean(R)) / np.std(R)
      vp = vp + alpha/(npop*sigma) * np.dot(N.T, A)
      ovnew = v2p(vp,params,sess)
      sess.run(uppar1,feed_dict={w1new:ovnew[0]})
      sess.run(uppar2,feed_dict={b1new:ovnew[1]})

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels}))
    
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)