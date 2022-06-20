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


def U_synth(u):
    N = u.shape[0]
    for i in range(1,N):
        if i == 1:
            uL = u[i:,:].T
            I = u[i-1,:]*np.eye(N-i)
            t = np.append(uL,I,axis=0)
            U = t
        
        else:
            Z = np.zeros((i-1,N-i))
            uL = u[i:,:].T
            I = u[i-1,:]*np.eye(N-i)
            t = np.append(Z,uL,axis=0)
            t = np.append(t,I,axis=0)    
            U = np.append(U,t, axis=1)
    return U

