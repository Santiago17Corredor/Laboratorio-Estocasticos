import numpy as np


def _normalize(v):
    return v / (np.linalg.norm(v) + 1e-10)
#clasifica

def classify(mic_norm: np.ndarray, mic_acov: np.ndarray):
    PATH_FM       = "src/dataSet/fmVector.txt"
    PATH_WN       = "src/dataSet/wnVector.txt"
    PATH_FM_ACOV  = "src/dataSet/fmAcov.txt"
    PATH_WN_ACOV  = "src/dataSet/wnAcov.txt"

    print("Cargando vectores de referencia...")
    fm_norm  = np.loadtxt(PATH_FM)
    wn_norm  = np.loadtxt(PATH_WN)
    fm_acov  = np.loadtxt(PATH_FM_ACOV)
    wn_acov  = np.loadtxt(PATH_WN_ACOV)

    size     = min(len(mic_norm), len(fm_norm), len(wn_norm))
    mic_norm = mic_norm[:size]
    fm_norm  = fm_norm[:size]
    wn_norm  = wn_norm[:size]

    size_acov = min(len(mic_acov), len(fm_acov), len(wn_acov))
    mic_acov  = mic_acov[:size_acov]
    fm_acov   = fm_acov[:size_acov]
    wn_acov   = wn_acov[:size_acov]

    mic_n = _normalize(mic_norm)
    fm_n  = _normalize(fm_norm)
    wn_n  = _normalize(wn_norm)

    dist_fm = np.mean(np.abs(mic_n - fm_n))
    dist_wn = np.mean(np.abs(mic_n - wn_n))

    print(f"Distancia con FM          : {dist_fm:.5f}")
    print(f"Distancia con Ruido Blanco: {dist_wn:.5f}")

    resultado = "FM" if dist_fm < dist_wn else "Ruido Blanco"
    print(f"Resultado: {resultado}")

    return resultado, dist_fm, dist_wn, mic_n, fm_n, wn_n, mic_acov, fm_acov, wn_acov