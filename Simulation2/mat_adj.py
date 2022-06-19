import numpy as np
import pandas as pd
import pickle as pkl

def Adj(N):
    A_gt = np.abs(np.random.randn(N,N))
    for i in range(N): A_gt[i,i] = 0
    A_gt = (A_gt + A_gt.T)/2
    return A_gt

def file(N):
    pkl.dump(Adj(N), open("Adj.pkl", "wb"))
    return "./Adj.pkl"
