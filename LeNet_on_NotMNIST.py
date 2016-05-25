# LeNet-5 Implementation, original parameter 98'
import tensorflow as tf
import numpy as np
import pickle

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10
batch_size = 16
num_channels = 1 # grayscale


def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

valid_size=10000
test_size=10000

C1_patch_size = 5
C3_patch_size = 5
C1_depth = 6
C3_depth = 16
C5_depth = 120
F6_depth = 84

graph = tf.Graph()

with graph.as_default():

    # Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset, shape=(valid_size,image_size,image_size,num_channels))
    tf_test_dataset = tf.constant(test_dataset, shape=(test_size,image_size,image_size,num_channels))

    # Variables.
    C1_weights = tf.Variable(tf.truncated_normal(
        [C1_patch_size, C1_patch_size, num_channels, C1_depth], stddev=0.1))
    C1_biases = tf.Variable(tf.zeros([C1_depth]))
    C3_weights = tf.Variable(tf.truncated_normal(
        [C3_patch_size, C3_patch_size, C1_depth, C3_depth], stddev=0.1))
    C3_biases = tf.Variable(tf.constant(1.0, shape=[C3_depth]))
    C5_weights_indim=(image_size//2-C3_patch_size+1)//2
    C5_weights = tf.Variable(tf.truncated_normal(
        [C3_depth * C5_weights_indim * C5_weights_indim , C5_depth], stddev=0.1))
    C5_biases = tf.Variable(tf.constant(1.0, shape=[C5_depth]))
    F6_weight = tf.Variable(tf.truncated_normal(
        [C5_depth,F6_depth]))
    F6_biases = tf.Variable(tf.constant(1.0, shape=[F6_depth]))
    G_weight = tf.Variable(tf.truncated_normal(
        [F6_depth,num_labels]
    ))
    G_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
    # layer4_weights = tf.Variable(tf.truncated_normal(
    #     [num_hidden, num_labels], stddev=0.1))
    # layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

    # Model.
    def model(data):
        ## To be honest I did not read the paper fully. What the heck is Gaussian connection anyway?

        # C1 6@28*28
        conv = tf.nn.conv2d(data, C1_weights, [1, 1, 1, 1], padding='SAME')
        conv = tf.nn.bias_add(conv, C1_biases)
        conv = tf.tanh(conv)
        # S2 6@14*14
        pool = tf.nn.max_pool(conv,[1,2,2,1],[1, 2, 2, 1], padding='SAME')
        pool = tf.tanh(pool)
        # C3 16@10*10
        # The mapping between S2 and C3 may not be true to LeNet-5 original design
        conv = tf.nn.conv2d(pool, C3_weights, [1, 1, 1, 1], padding='VALID')
        conv = tf.nn.bias_add(conv, C3_biases)
        conv = tf.tanh(conv)
        # S4 16@5*5
        pool = tf.nn.max_pool(conv,[1,2,2,1],[1,2,2,1],padding='SAME')
        pool = tf.tanh(pool)
        # C5 120 implemented as matmul
        shape = pool.get_shape().as_list()
        reshape= tf.reshape(pool,[shape[0], -1]) # Linear list for full connectivity
        conv = tf.matmul(reshape, C5_weights) + C5_biases
        conv = tf.tanh(conv)
        # F6 84
        F6 = tf.matmul(conv, F6_weight) + F6_biases
        F6 = tf.tanh(F6)
        #
        output= tf.matmul(F6,G_weight) + G_biases
        return output

    # Training computation.
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)


    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

num_steps = 6001

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(
                valid_prediction.eval(), valid_labels))
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))