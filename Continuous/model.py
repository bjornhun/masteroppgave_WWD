import tensorflow as tf
from preprocessing import *
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, confusion_matrix

model_name = "single_layer_gru_20epochs_40coeffs_doubleneg"

# Set parameters
n_inputs = 40
n_neurons = 128
n_outputs = 2
learning_rate = 0.001
n_epochs = 20
batch_size = 256
threshold = .99
mean, std = pickle.load(open("pickles/stats.p", "rb"))

# The two following functions were copied from this article:
# https://danijar.com/variable-sequence-lengths-in-tensorflow/
def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length

def last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant

def pad(data):
    return pad_sequences(data, padding="post", dtype=np.float64)

# Define placeholders
X = tf.placeholder(tf.float32, [None, None, n_inputs])
y = tf.placeholder(tf.int32, [None])
initial_state = tf.placeholder(tf.float32, [None, n_neurons])

seq_length = length(X)

cell = tf.contrib.rnn.GRUCell(n_neurons)
outputs, states = tf.nn.dynamic_rnn(cell=cell,
                                    inputs=X,
                                    dtype=tf.float32,
                                    initial_state=initial_state,
                                    sequence_length=seq_length)

# Compute wake word probabilities from RNN outputs
last = last_relevant(outputs, seq_length)
weight = tf.Variable(
    tf.truncated_normal([n_neurons, n_outputs], stddev=0.1))
bias = tf.Variable(tf.constant(0.1, shape=[n_outputs]))
logits = tf.matmul(last, weight) + bias
last_probs = tf.nn.softmax(logits)[:,1]

stacked_rnn_outputs = tf.reshape(outputs, [-1, n_neurons])
stacked_outputs = tf.matmul(stacked_rnn_outputs, weight) + bias
outputs = tf.reshape(stacked_outputs, [-1, tf.reduce_max(seq_length), n_outputs])
wakeword_probs = tf.nn.softmax(outputs)[0,:,1] # Wake word probabilities for each timestep

# Optimize
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

# Compute accuracy
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Initialize global variables and saver
init = tf.global_variables_initializer()
saver = tf.train.Saver()