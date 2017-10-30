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

def main(_):

  solution = tf.constant([0.5, 0.1, -0.3],dtype=tf.float32)
  w = tf.Variable(np.random.randn(3),dtype=tf.float32)
  w_try = tf.Variable(np.zeros(3),dtype=tf.float32)
  reward = -tf.reduce_sum(tf.square(solution - w))
  reward_try = -tf.reduce_sum(tf.square(solution - w_try))
  w_tryarr = tf.placeholder(tf.float32, [3])
  uptry = tf.assign(w_try,w_tryarr)


  with tf.Session() as sess:
    # global sess
    sess.run(tf.global_variables_initializer())
    for i in range(300):     
      if i % 20 == 0:
        print('iter %d. w: %s, solution: %s, reward: %f' % 
              (i, sess.run(w), sess.run(solution), sess.run(reward)))
      N = np.random.randn(npop, 3)
      if i==0:
        print(N)
      R = np.zeros(npop)
      for j in range(npop):
        w_tryar = sess.run(w) + sigma*N[j] # jitter w using gaussian of sigma 0.1
        if j==0 and i==0:
          print("w_try j0:",w_try)
        if j==1 and i==0:
          print("w_try j1:",w_try)
          print("N[j]:",N[j])
          print("w:",sess.run(w))
        # uptry =  tf.assign(w_try,w_tryarr)
        sess.run(uptry,feed_dict={w_tryarr: w_tryar})
        R[j] = sess.run(reward_try) # evaluate the jittered version
        # print(R[j])
      if i==0:
        print(R)


      # standardize the rewards to have a gaussian distribution
      A = (R - np.mean(R)) / np.std(R)
      # perform the parameter update. The matrix multiply below
      # is just an efficient way to sum up all the rows of the noise matrix N,
      # where each row N[j] is weighted by A[j]
      kk = sess.run(w)

      pp = alpha/(npop*sigma) * np.dot(N.T, A)

      wnew = sess.run(w) + alpha/(npop*sigma) * np.dot(N.T, A)
      upw = tf.assign(w,wnew)
      sess.run(upw)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)