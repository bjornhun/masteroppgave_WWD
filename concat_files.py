import os
from scipy.io import wavfile
import numpy as np
import pickle
import random

datapath = "data/test/"
targetpath = "data/long/"
wakeword = "wakeword-heywebex"
n_neg = 20
n_pos = 5

def concat_files(fname):
    '''Concatenates test files to produce longer file containing 
    multiple utterances. Writes result to new file.'''
    files = os.listdir(datapath)
    positives = [f for f in files if f.startswith(wakeword)]
    negatives = [f for f in files if not f.startswith(wakeword)]
    
    pos_samples = random.sample(positives, n_pos)
    neg_samples = random.sample(negatives, n_neg)

    files = pos_samples + neg_samples
    random.shuffle(files)

    speech = []
    for f in files:
        fs, x = wavfile.read(datapath + f)
        if len(x.shape) > 1:
            x = x[:, 0] # Only use one channel
        for val in x:
            speech.append(val)
    speech = np.asarray(speech, dtype=np.int16)

    wavfile.write(fname, fs, speech)

if __name__ == '__main__':
    os.makedirs(targetpath)
    concat_files(targetpath+"concat_file1.wav")
    concat_files(targetpath+"concat_file2.wav")
    concat_files(targetpath+"concat_file3.wav")