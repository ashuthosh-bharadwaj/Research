import numpy as np
import pandas as pd
import pickle as pkl
 
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

def U_synth2(u):
    N = u.shape[0]
    for i in range(1,N+1):
        if i == 1:
            u_there = (u[i-1,:]*(np.eye(N))[:,i-1]).reshape((N,1))
            uL = u[i:,:].T
            I = u[i-1,:]*np.eye(N-i)
            t = np.append(uL,I,axis=0)
            t = np.append(u_there,t,axis=1)
            U = t
        
        elif i == N:
            t = (u[i-1,:]*(np.eye(N))[:,i-1]).reshape((N,1))
            U = np.append(U,t,axis=1)

        else:
            u_there = (u[i-1,:]*(np.eye(N))[:,i-1]).reshape((N,1))
            Z = np.zeros((i-1,N-i))
            uL = u[i:,:].T
            I = u[i-1,:]*np.eye(N-i)
            t = np.append(Z,uL,axis=0)
            t = np.append(t,I,axis=0)
            t = np.append(u_there,t,axis=1)    
            U = np.append(U,t, axis=1)

    return U

def Make(sd,N):
    np.random.randn(sd)
    A_gt = np.abs(np.random.randn(N,N))
    for i in range(N): A_gt[i,i] = 0
    A_gt = (A_gt + A_gt.T)/2
    D = np.diag(np.dot(A_gt, np.ones(N,)))
    return D, A_gt

def Store(sd,N):
    d, a = Make(sd,N)
    pkl.dump(d, open("./Data/Deg.pkl", "wb"))
    pkl.dump(a, open("./Data/Adj.pkl", "wb"))
    return "./Data/Deg.pkl" , "./Data/Adj.pkl"

def vectorizer(M):
    N = M.shape[0]
    vec = M[1:,0]
    for i in range(1,N):  
        vec = np.append(vec, M[i+1:,i], axis = 0)
    return vec

def vectorizer2(M):
    N = M.shape[0]
    vec = M[:,0]
    for i in range(1,N):  
        vec = np.append(vec, M[i:,i], axis = 0)
    return vec

def matricizer(vec):
    N = vec.shape[0]
    N = int((1+np.sqrt(1+8*N))//2)
    M = np.zeros((N,N))
    k1 = 0
    k2 = N-1
    for i in range(N):
        s = N - i-2
        M[i+1:,i] = vec[k1:k2]
        k1 = k2 
        k2 += s
    return M + M.T

def matricizer2(vec):
    N = vec.shape[0]
    N = int((-1+np.sqrt(1+8*N))//2)
    M = np.zeros((N,N))
    k1 = 0
    k2 = N
    for i in range(N):
        s = N-i-1
        M[i:,i] = vec[k1:k2]
        k1 = k2 
        k2 += s
    return M + M.T - np.diag(np.diag(M))

def SNR_adder(sig, snr):
    noi = np.random.randn(*np.shape(sig))
    es = np.sum(sig**2)
    en = np.sum(noi**2)
    a = np.sqrt(es/((10**snr)*en))
    return sig + a*noi

def adja(L):
    A = -L
    N = A.shape[0]
    for i in range(N): A[i,i] =0
    return A