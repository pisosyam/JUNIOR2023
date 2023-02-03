import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
from celluloid import Camera
# import warnings

from scipy.linalg import cholesky
from scipy import sparse
from scipy.sparse.linalg import spsolve


def baseline_arPLS(y, lam=1e4, ratio=0.05, itermax=100):
    y = y.astype(np.float)
    N = len(y)
    D = sparse.eye(N, format='csc')
    D = D[1:] - D[:-1]
    D = D[1:] - D[:-1]
    H = lam * D.T * D
    w = np.ones(N)
    for i in range(itermax):
        W = sparse.diags(w, 0, shape=(N, N))
        WH = sparse.csc_matrix(W + H)
        C = sparse.csc_matrix(cholesky(WH.todense()))
        z = spsolve(C, spsolve(C.T, w * y.astype(np.float64)).astype(np.float64))
        d = y - z
        dn = d[d < 0]
        m = np.mean(dn)
        s = np.std(dn)
        wt = 1. / (1 + np.exp(2 * (d - (2 * s - m)) / s))
        if np.linalg.norm(w - wt) / np.linalg.norm(w) < ratio:
            break
        w = wt
    return z


files = [18, 19, 30, 31, 35]

for file in files:
    df_P = pd.read_excel(f"final/{file}.xlsx", header=None)
    Z_P = df_P.iloc[1:, 9].to_numpy()

    fig, ax = plt.subplots()
    plt.plot(Z_P, label='line', color='black')
    plt.xlabel('Компенсационное напряжение, см2/(В·c)')
    plt.ylabel('Ионный ток, ЕМЗР')
    fig.savefig(f'final/final_results/{file}_raw.png')
    fig.clear(True)

    baseline = baseline_arPLS(Z_P)
    plt.plot(Z_P - baseline, label='processed_line', color='black')
    plt.xlabel('Компенсационное напряжение, см2/(В·c)')
    plt.ylabel('Ионный ток, ЕМЗР')
    fig.savefig(f'final/final_results/{file}_processed.png')
