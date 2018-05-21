import tensorflow as tf
from preprocessing import *
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, confusion_matrix

model_name = "single_layer_gru_20epochs_40coeffs_doubleneg"

if __name__ == '__main__':
    train_model = False
    evaluate_model = False
    plot_preds = True

else:
    train_model = False
    evaluate_model = False
    plot_preds = False  

# Set parameters
n_inputs = 40
n_neurons = 128
n_outputs = 2
n_timesteps = 149
learning_rate = 0.001
n_epochs = 20
batch_size = 128
threshold = .99
mean, std = pickle.load(open("pickles/stats.p", "rb"))

def batchify(X, y):
    n_data = len(y)
    n_batches = int(n_data/batch_size)

    indices = list(range(n_data))
    random.shuffle(indices)
    X = X[indices]
    y = y[indices] 

    X_batches = np.array_split(X[:-(n_data%batch_size)], n_batches)
    y_batches = np.array_split(y[:-(n_data%batch_size)], n_batches)

    return X_batches, y_batches

# Define placeholders
X = tf.placeholder(tf.float32, [None, None, n_inputs])
y = tf.placeholder(tf.int32, [None])

cell = tf.contrib.rnn.GRUCell(n_neurons)
outputs, states = tf.nn.dynamic_rnn(cell=cell,
                                    inputs=X,
                                    dtype=tf.float32)

# Compute wake word probabilities from RNN outputs
weight = tf.Variable(
    tf.truncated_normal([n_neurons, n_outputs], stddev=0.1))
bias = tf.Variable(tf.constant(0.1, shape=[n_outputs]))

stacked_rnn_outputs = tf.reshape(outputs, [-1, n_neurons])
stacked_outputs = tf.matmul(stacked_rnn_outputs, weight) + bias
outputs = tf.reshape(stacked_outputs, [-1, n_timesteps, n_outputs])
logits = outputs[:,-1, :] # Wake word probability for last timestep
wakeword_prob = tf.nn.softmax(outputs)[:,-1,1] # Wake word probabilities for last timestep

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