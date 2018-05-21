from preprocessing import *
from model import *
import pyaudio
import winsound 
import time

fs = 48000
framesize = 12000
n_secs = 20

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=fs, input=True, frames_per_buffer=framesize)

def record():
    data = stream.read(framesize)
    raw_speech = np.fromstring(data, dtype=np.int16)
    return normalize(get_mfcc(raw_speech, fs), mean, std)

with tf.Session() as sess:
    saver.restore(sess, "./models/" + model_name + ".ckpt")
    probs = []
    st = np.zeros((1, n_neurons))
    start = time.time()
    count = 0
    print("Recording...")
    while True:
        coeff = record()
        st, wp = sess.run([states, wakeword_probs], feed_dict={X: [coeff], initial_state: st})

        for val in wp:
            probs.append(val)
        
        detections = np.asarray(probs) > threshold

        if True in (wp > threshold):
            count+=1
            print("Wake word detected #", count)
            st = np.zeros((1, n_neurons))

        if (time.time() - start > n_secs):
            probs = np.asarray(probs)
            x = np.linspace(0, len(probs)-1, len(probs))
            plt.plot(x, probs)
            plt.plot(x[detections], probs[detections], 'go')
            plt.show()
            stream.stop_stream()
            stream.close()
            p.terminate()
            break