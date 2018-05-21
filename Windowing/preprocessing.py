import numpy as np
from scipy.io import wavfile
from python_speech_features import mfcc
import os
import random
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
from keras.preprocessing.sequence import pad_sequences

wakeword = "wakeword-heywebex"
train_path = "../data/train/"
test_path = "../data/test/"
val_path = "../data/val/"

length = int(48000 * 1.5)

def get_mfcc(samples, sample_rate):
    mfccs = mfcc(samples, numcep=40, samplerate=sample_rate, nfft=2048, highfreq=8000, nfilt=40)
    return mfccs

def preprocess(folder, filenames):
    '''Reads all data from filepath and gets MFCC coefficients.
    Returns MFCC data and corresponding labels for all files.'''
    random.shuffle(filenames)
    num_of_files = len(filenames)
    X = []
    y = []
    count = 1
    skipped = 0
    for f in filenames:
        sample_rate, samples = wavfile.read(folder + f)
        if len(samples) < length or np.average(np.abs(samples)) == 0:
            skipped += 1 #remove files shorter than 1.5 s or containing only silence
            print(str(skipped) + " files skipped")
            continue
        if len(samples.shape) > 1:
            samples = samples[:, 0] # Only use one channel
        
        if f.startswith(wakeword):
            stripped = strip_silence(samples)
            coeff = get_mfcc(stripped, sample_rate)
            X.append(coeff)
            y.append(1)
            
        else:
            stripped = strip_silence(samples)
            coeff = get_mfcc(stripped, sample_rate)
            X.append(coeff)
            y.append(0)

            rand_index = random.randint(0,len(samples)-length)
            coeff = get_mfcc(samples[rand_index:rand_index+length], sample_rate)
            X.append(coeff)
            y.append(0)

        if count % 100 == 0:
            print(str(count) + "/" + str(num_of_files) + " files preprocessed.")
        count+=1
    return np.asarray(X), np.asarray(y)

def normalize(data, mean, std):
    data -= mean
    data /= std
    return data

def get_filepaths():
    train = [f for f in os.listdir(train_path)]
    test = [f for f in os.listdir(test_path)]
    val = [f for f in os.listdir(val_path)]
    return train, test, val

def get_stats(data):
    mean = np.mean(np.concatenate(X_train).ravel())
    std = np.std(np.concatenate(X_train).ravel())
    return mean, std    

def strip_silence(x, step_size=4800):
    n_samples = len(x)
    hi_avg = 0
    hi_index = 0
    for i in range(0, n_samples-length, step_size):
        avg = np.average(np.abs(x[i:i+length]))
        if avg > hi_avg:
            hi_avg = avg
            hi_index = i
    return x[hi_index:hi_index+length]

def plot_mfcc(coeff):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    fig, ax = plt.subplots()
    coeff = np.swapaxes(coeff, 0 ,1)
    cax = ax.imshow(coeff, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
    ax.set_title('MFCC')
    plt.show()

if __name__ == '__main__':
    train, test, val = get_filepaths()

    X_train, y_train = preprocess(train_path, train)
    X_test, y_test = preprocess(test_path, test)
    X_val, y_val = preprocess(val_path, val)

    mean, std = get_stats(X_train)

    X_train = normalize(X_train, mean, std)
    X_test = normalize(X_test, mean, std)
    X_val = normalize(X_val, mean, std)

    pickle.dump((X_train, y_train), open("pickles/train.p", "wb"))
    pickle.dump((X_test, y_test), open("pickles/test.p", "wb"))
    pickle.dump((X_val, y_val), open("pickles/val.p", "wb"))
    pickle.dump((mean, std), open("pickles/stats.p", "wb"))