import os
import numpy as np
import random

wakeword = "wakeword-heywebex"
test_ratio = .1
val_ratio = .1

def train_test_val_split(filepath):
    '''Randomizes files and splits them into three separate sets.'''
    files = os.listdir(filepath)
    random.shuffle(files)
    positives = [f for f in files if f.startswith(wakeword)]
    negatives = [f for f in files if not f.startswith(wakeword)]

    n_test_pos = int(test_ratio*len(positives))
    n_test_neg = int(test_ratio*len(negatives))
    n_val_pos = int(val_ratio*len(positives))
    n_val_neg = int(val_ratio*len(negatives))

    train = []
    test = []
    val = []
    
    test.extend([positives.pop() for i in range(n_test_pos)])
    test.extend([negatives.pop() for i in range(n_test_neg)])
    val.extend([positives.pop() for i in range(n_val_pos)])
    val.extend([negatives.pop() for i in range(n_val_neg)])
    train.extend(positives)
    train.extend(negatives)

    return train, test, val

if __name__ == '__main__':
    train, test, val = train_test_val_split("data/")

    # Create subfolders
    os.makedirs("data/train")
    os.makedirs("data/test")
    os.makedirs("data/val")
    
    # Move files into subfolders
    for f in train:
        os.rename("data/" + f, "data/train/" + f)
    for f in test:
        os.rename("data/" + f, "data/test/" + f)
    for f in val:
        os.rename("data/" + f, "data/val/" + f)