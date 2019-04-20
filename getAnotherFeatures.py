# inspired by this paper's code 
# Unsupervised speech recognition through spike-timing-dependent plasticity in a convolutional spiking neural network


from python_speech_features import fbank
import numpy as np 
import scipy.io.wavfile as wav 
import os 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import pandas as pd 
from sklearn.utils import shuffle


def read_the_csv():

    ''' Read the csv file and extract the MFSCs features'''
    n_bands = 41
    n_frames = 40
    overlap = 0.5

    datasets = pd.read_csv('data.csv')
    print(len(datasets))
    
    datasets = shuffle(datasets)
    n_samples = len(datasets)

    feats = np.empty((n_samples, n_bands * n_frames))
    labels = np.empty((n_samples,), dtype=np.uint8)
    
    
    for i in range(n_samples):
        label = datasets['label'].iloc[i]
        file = datasets['filename'].iloc[i]
        labels[i] = np.uint8(label)

        rate , sig = wav.read(file)
        duration = sig.size / rate
        winlen = duration /(n_frames * (1 - overlap) + overlap)
        winstep = winlen * (1 - overlap)

        feat, energy = fbank(sig, rate, winlen, winstep, 
        nfilt=n_bands, nfft=4096, winfunc=np.hamming)

        feat = np.log(feat)
        
        feats[i] = feat[:n_frames].flatten()
    
    feats = normalize(feats, norm = 'l2', axis=1)

    np.random.seed(42)
    np.random.shuffle(feats)
    p = np.random.permutation(n_samples)
    feats, labels[p], labels[p]

    n_train_sample = int(n_samples * 0.7)

    train_set = (feats[:n_train_sample], labels[:n_train_sample])
    test_set = (feats[n_train_sample:], labels[n_train_sample:])


    return train_set, test_set

train, test = read_the_csv()
