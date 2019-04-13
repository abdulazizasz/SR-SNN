import os 
from collections import Counter
import csv
import librosa
import numpy as np
import pandas as pd 

# Inspired by https://towardsdatascience.com/extract-features-of-music-75a3f9bc265d
# https://github.com/moebg/spoken-digit-recognition/blob/master/src/spoken_digit.py

def build_features():
    # generating a dataset
    header = 'filename'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()

    file = open('./data.csv', 'w', newline = '')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
    classes = [0,1,2,3,4,5,6,7,8,9]
    dic = {}
    for label in classes:
        for file in os.listdir(f'./datasets/{label}'):
            filename = f'./datasets/{label}/{file}'
            x , sr = librosa.load(filename)
            mfcc = librosa.feature.mfcc(x , sr)
            # Check the frames. Make sure all have 40 frames 
            if mfcc.shape[1] > 20:
                mfcc = mfcc[:, 0:20]
            else:
                mfcc = np.pad(mfcc, ((0, 0), (0, 20 - mfcc.shape[1])),
                               mode='constant', constant_values=0)
           
            inputs = f' {filename} '
            for feature in mfcc:
                inputs += f' {np.mean(feature)} '
            inputs += f' {label} '
            file = open('./data.csv','a', newline = '')
            with file:
                writer = csv.writer(file)
                writer.writerow(inputs.split())


if __name__ == "__main__":
    build = True
    if build:
        build_features()
    data = pd.read_csv('./data.csv')
    
    labels = data.iloc[:,-1]
    features = np.array(data.iloc[:,1:-1])
    print(features.shape[0])      

