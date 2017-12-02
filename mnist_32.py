import datetime
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#getting the start time of the program.
startTime = datetime.datetime.now()

##
 # returns pixels values and labels of the training data
 # @param  size:  number of images
 # @return x_train : pixel values of images
 # @return y_train : labels for corresponding images
##

def TRAIN_SIZE(size):
	x_train = mnist.train.images[:size,:]
	y_train = mnist.train.labels[:size,:]
	return x_train, y_train


##
 # returns pixels values and labels of the testing data
 # @param  size:  number of images
 # @return x_train : pixel values of images
 # @return y_train : labels for corresponding images
##

def TEST_SIZE(size):
	x_test = mnist.test.images[:size,:]
	y_test = mnist.test.labels[:size,:]
	return x_test, y_test



#Placeholder defined to hold the pixel values of the 28*28 pixel images and output label values#
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])


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
y = tf.nn.softmax(tf.matmul(y2, W3) + b3)



# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
# train_step = tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

for i in range(10000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  if i%1000 == 0:
#         correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#         accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        values = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
       	print('Accuracy :: ',values)
        print('For :: ', i)
        print('Time elasped :: ', datetime.datetime.now())


tf = datetime.datetime.now()
te = tf - ts

print('Start Time :: ',ts)
print('Finish Time :: ',tf)
print('Total Time Elapsed :: ',te)