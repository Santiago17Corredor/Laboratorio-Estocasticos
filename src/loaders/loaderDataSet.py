import src.processing.process as process
import numpy as np


def _loaderDataSet():

    pathsFM = process.obtainPaths("src/data/FM")
    pathsWN = process.obtainPaths("src/data/WN")

    print(f"FM: {len(pathsFM)} archivos")
    print(f"WN: {len(pathsWN)} archivos")

    resultsFM = []
    resultsWN = []

    for f in pathsFM:
        process.fillArray(f, resultsFM)

    for f in pathsWN:
        process.fillArray(f, resultsWN)

    avg_fm = process.calcAvgVector(resultsFM)
    avg_wn = process.calcAvgVector(resultsWN)

    print(f"\nNorma promedio FM: {np.mean(avg_fm['norm']):.5f}")
    print(f"Norma promedio WN: {np.mean(avg_wn['norm']):.5f}")

    return avg_fm, avg_wn


def _saveData(avg_fm, avg_wn):
    np.savetxt("src/dataSet/fmVector.txt",  avg_fm["norm"])
    np.savetxt("src/dataSet/wnVector.txt",  avg_wn["norm"])
    np.savetxt("src/dataSet/fmAcov.txt",    avg_fm["acov"])
    np.savetxt("src/dataSet/wnAcov.txt",    avg_wn["acov"])


def train():
    print("Iniciando entrenamiento...")
    avg_fm, avg_wn = _loaderDataSet()
    _saveData(avg_fm, avg_wn)
    print("Entrenamiento completado.")

train()