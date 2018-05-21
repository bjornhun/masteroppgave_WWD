from model import *

X_train, y_train = pickle.load(open("pickles/train.p", "rb"))
X_val, y_val = pickle.load(open("pickles/val.p", "rb"))

with tf.Session() as sess:
    init.run()

    print("Start time:", datetime.now().time())
    for epoch in range(n_epochs):
        X_batches, y_batches = batchify(X_train, y_train)
        n_batches = len(X_batches)
        avg_loss = 0.
        for i in range(n_batches):
            X_batch, y_batch = X_batches[i], y_batches[i]

            _, c, init_state, logs = sess.run([training_op, loss, states, logits],
                                        feed_dict={X: X_batch,
                                                    y: y_batch})
            avg_loss += c / n_batches
        print("Epoch:", '%04d' % (epoch+1),
                "loss=", "{:.9f}".format(avg_loss),
                "Val accuracy:", accuracy.eval({X: X_val,
                                                y: y_val}),
                "Time finished:", datetime.now().time())


    print("Optimization Finished!")
    save_path = saver.save(sess, "./models/" + model_name + ".ckpt")