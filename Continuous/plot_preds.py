from model import *
import time

with tf.Session() as sess:
    saver.restore(sess, "./models/" + model_name + ".ckpt")

    rec_dict = {"podcast":  [],
                "podcast2": [],
                "concat_file1": [190, 1630, 1956, 5641, 6205],
                "concat_file2": [689, 3087, 5351, 5637, 6253],
                "concat_file3": [463, 2784, 3095, 5583, 6760],
                }

    for fname in rec_dict.keys():
        if not os.path.exists("plots/" + model_name):
            os.makedirs("plots/" + model_name)
        fs, x = wavfile.read("../data/long/" + fname + ".wav")
        init_state = np.zeros(shape=((1, n_neurons)))
        ww_times = rec_dict[fname]

        probs=[]
        n_detections = 0
        step = 12000
        start_time = time.time()
        for i in range(0, len(x)-step, step):
            coeff = normalize(get_mfcc(x[i:i+step], fs), mean, std)
            init_state, wp = sess.run([states, wakeword_probs], feed_dict={X: [coeff], initial_state: init_state})
            detection = False
            for val in wp:
                probs.append(val)
                if val > threshold:
                    init_state = np.zeros(shape=((1, n_neurons)))
                    detection = True
            if detection:
                n_detections += 1
        time_passed = time.time() - start_time
        print(fname + ": " + str(n_detections) + " detections, time passed: " + str(time_passed))

        probs = np.asarray(probs)

        detections = probs > threshold

        x = np.linspace(0, len(probs)-1, len(probs))
        plt.plot(x[detections], probs[detections], 'go')
        plt.plot(x, probs)
        for t in ww_times:
            plt.axvline(x=t, color="red")
        if len(probs) < 96000:
            ticks = np.arange(0, len(probs), 960)
            plt.xticks(ticks, [int(i/96) for i in ticks])
            plt.xlabel("Time [seconds]")
        else:
            ticks = np.arange(0, len(probs), 96*60*5)
            plt.xticks(ticks, [int(i/(96*60)) for i in ticks])
            plt.xlabel("Time [minutes]")
        plt.ylabel("Wake word probability")
        plt.tight_layout()
        plt.savefig("plots/" + model_name + "/" + fname)
        plt.clf()