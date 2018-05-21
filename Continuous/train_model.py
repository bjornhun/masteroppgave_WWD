from model import *

X_train, y_train = pickle.load(open("pickles/train.p", "rb"))
X_val, y_val = pickle.load(open("pickles/val.p", "rb"))
X_val = pad(X_val)

def batchify(X, y):
    n_data = len(y)
    n_batches = int(n_data/batch_size)

    indices = list(range(n_data))
    random.shuffle(indices)
    X = X[indices]
    y = y[indices] 

    X_batches = np.array_split(X[:-(n_data%batch_size)], n_batches)
    y_batches = np.array_split(y[:-(n_data%batch_size)], n_batches)

    X_batches = [pad(batch) for batch in X_batches]
    return X_batches, y_batches

with tf.Session() as sess:
    init.run()
    init_state = np.zeros(shape=((batch_size, n_neurons)))
    val_inx = np.random.randint(low=0, high=batch_size, size=len(y_val))
    
    print("Start time:", datetime.now().time())
    for epoch in range(n_epochs):
        X_batches, y_batches = batchify(X_train, y_train)
        n_batches = len(X_batches)
        avg_loss = 0.
        for i in range(n_batches):
            X_batch, y_batch = X_batches[i], y_batches[i]

            _, c, init_state, logs = sess.run([training_op, loss, states, logits],
                                        feed_dict={X: X_batch,
                                                    y: y_batch,
                                                    initial_state: init_state})
            avg_loss += c / n_batches
        print("Epoch:", '%04d' % (epoch+1),
                "loss=", "{:.9f}".format(avg_loss),
                "Val accuracy:", accuracy.eval({X: X_val,
                                                y: y_val,
                                                initial_state: init_state[val_inx]}),
                "Time finished:", datetime.now().time())

    print("Optimization Finished!")
    pickle.dump((init_state), open("pickles/last_state.p", "wb"))
    save_path = saver.save(sess, "./models/" + model_name + ".ckpt")