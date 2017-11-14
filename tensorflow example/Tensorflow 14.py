import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size, activation_function=None):
    #add one more layer and return the output of this layer
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')   #in_size <= rows    out_size <= cols
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)  #biases 不為零 所以+0.1
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

#Make up some real data
x_data = np.linspace(-1,1,300)[:,np.newaxis] #300 rows   1 col
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

#define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

#add hidden layer
l1 = add_layer(xs, 1, 10, activation_function= tf.nn.relu)
#add output layer
predition = add_layer(l1, 10, 1, activation_function=None)

#the error between predicition and real data
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - predition), reduction_indices = [1]))

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#inportant step
init = tf.initialize_all_variables()
sess = tf.Session()
writer = tf.summary.FileWriter("logs/",sess.graph)
sess.run(init)

fig = plt.figure()#產生圖片框
ax = fig.add_subplot(1,1,1)#連續畫圖
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

for i in range(1000):
    sess.run(train_step, feed_dict = {xs:x_data, ys:y_data})
    if i % 50 == 0:
        #print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        predition_value = sess.run(predition, feed_dict={xs:x_data})
        lines = ax.plot(x_data, predition_value, 'r-', lw=5)  #用寬度為5(lw=5)的紅線畫線('r-')
        plt.pause(0.1)
