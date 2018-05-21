from model import *

X_test, y_test = pickle.load(open("pickles/test.p", "rb"))

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim(0,1)
    plt.tight_layout()

with tf.Session() as sess:
    saver.restore(sess, "./models/" + model_name + ".ckpt")

    acc, probs = sess.run([accuracy, wakeword_prob], feed_dict={X: X_test,
                                            y: y_test})

    preds = np.rint(probs)
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, preds)
    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)

    print("Test accuracy:", acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F-score:", fscore)
    print("Support:", support)
    print("Confusion matrix:", confusion_matrix(y_test, preds))
    
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
    plt.show()