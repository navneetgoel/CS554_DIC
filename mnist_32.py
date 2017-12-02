import datetime
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#getting the start time of the program.
startTime = datetime.datetime.now()

#Placeholder defined to hold the pixel values of the 28*28 pixel images and output label values#
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])


#
#-------------LAYER 1-----------
# weights and baises are declared.
#
W1 = tf.Variable(tf.random_normal([784, 100]))
b1 = tf.Variable(tf.random_normal([100,]))
y1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

#
#-------------LAYER 2-----------
# weights and baises are declared.
#
W2 = tf.Variable(tf.random_normal([100,10]))
b2 = tf.Variable(tf.random_normal([10,]))
y2 = tf.nn.sigmoid(tf.matmul(y1, W2) + b2)

#
#-------------OUTPUT LAYER-----------
# weights and baises are declared.
#
W3 = tf.Variable(tf.random_normal([10,10]))
b3 = tf.Variable(tf.random_normal([10,]))
prediction = tf.nn.softmax(tf.matmul(y2, W3) + b3)

# Calculating cross entropy to find the accuracy of the model
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

#
# Variable are defined to initialize the model
# Total Training images in dataset = 55000
# Total Test images in dataset = 10000
#
x_test, y_test = TEST_SIZE(10000)
LEARNING_RATE = 0.01
TRAIN_STEPS = 10000
BATCH_SIZE = 100

# optimizing the values of weights and baises to reduce cost.
training = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initializing the global variables
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

#
# Training and testing the model 
# Also calculating various metrics for the model
# Accuracy, Elapsed Time, Cost
#

for i in range(TRAIN_STEPS):
  x_train, y_train = mnist.train.next_batch(BATCH_SIZE)
  sess.run(training, feed_dict={x: x_train, y: y_train})
  if i%1000 == 0:
    values = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
    print('Accuracy :: ',values)
    print('For :: ', i)
    print('Time elasped :: ', datetime.datetime.now())


tf = datetime.datetime.now()
te = tf - ts

print('Start Time :: ',ts)
print('Finish Time :: ',tf)
print('Total Time Elapsed :: ',te)