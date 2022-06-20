import numpy as np
import pandas as pd
import pickle as pkl

def Make(N):
    A_gt = np.abs(np.random.randn(N,N))
    for i in range(N): A_gt[i,i] = 0
    A_gt = (A_gt + A_gt.T)/2
    D = np.diag(np.dot(A_gt, np.ones(N,)))
    return D, A_gt

def Store(N):
    d, a = Make(N)
    pkl.dump(d, open("Deg.pkl", "wb"))
    pkl.dump(a, open("Adj.pkl", "wb"))
    return "./Deg.pkl" , "./Adj.pkl"