import os
from statsmodels.tsa.stattools import acovf
import numpy as np
import librosa
 #como se calcula el audio
 
def obtainPaths(path):
    return [os.path.join(path, file) for file in os.listdir(path)]
 
def loadAudio(file_path, sr=44100):
    y, _ = librosa.load(file_path, sr=sr, dtype=np.float64)
    return y
 
def calcAutocovariance(y):
    return acovf(y, fft=True, demean=True).astype(np.float64)
 
def calcFourier(acov):
    return np.fft.fft(acov)
 
def calcNorm(fourier):
    n = len(fourier)
    return np.abs(fourier)
 
def fillArray(f, results):
    y = loadAudio(f)
 
    acov    = calcAutocovariance(y)
    fourier = calcFourier(acov)
    norm    = calcNorm(fourier)
 
    results.append([acov, fourier, norm])
 
def calcAvgVector(results):
    acovs, fouriers, norms = zip(*results)
 
    return {
        "acov":    np.mean(acovs,    axis=0),
        "fourier": np.mean(fouriers, axis=0),
        "norm":    np.mean(norms,    axis=0),
    }
 
