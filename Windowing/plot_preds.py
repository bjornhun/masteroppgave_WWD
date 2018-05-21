from model import *
import time 

with tf.Session() as sess:
    saver.restore(sess, "./models/" + model_name + ".ckpt")

    rec_dict = {"podcast": [],
                "podcast2": [],
                "concat_file1": [190//24, 1630//24, 1956//24, 5641//24, 6205//24],
                "concat_file2": [689//24, 3087//24, 5351//24, 5637//24, 6253//24],
                "concat_file3": [463//24, 2784//24, 3095//24, 5583//24, 6760//24],
                }

    for fname in rec_dict.keys():
        if not os.path.exists("plots/" + model_name):
            os.makedirs("plots/" + model_name)
        fs, x = wavfile.read("../data/long/" + fname + ".wav")
        init_state = np.zeros(shape=((1, n_neurons)))
        ww_times = rec_dict[fname]

        step = 12000
        probs=[0]*(length//step)
        start_time = time.time()
        for i in range(0, len(x)-length, step):
            coeff = normalize(get_mfcc(x[i:i+length], fs), mean, std)
            init_state, wp = sess.run([states, wakeword_prob], feed_dict={X: [coeff]})
            probs.append(wp)
        time_passed = time.time() - start_time
        print(fname + ": time passed: " + str(time_passed))

        probs = np.asarray(probs)

        detections = probs > threshold

        x = np.linspace(0, len(probs)-1, len(probs))
        plt.plot(x[detections], probs[detections], 'go')
        plt.plot(x, probs)
        for t in ww_times:
            plt.axvline(x=t, color="red")
        if len(probs) < 1800:
            ticks = np.arange(0, len(probs), 40)
            plt.xticks(ticks, [int(i/4) for i in ticks])
            plt.xlabel("Time [seconds]")
        else:
            ticks = np.arange(0, len(probs), 4*60*5)
            plt.xticks(ticks, [int(i/(4*60)) for i in ticks])
            plt.xlabel("Time [minutes]")
        plt.ylabel("Wake word probability")
        plt.tight_layout()
        plt.savefig("plots/" + model_name + "/" + fname)

        plt.clf()