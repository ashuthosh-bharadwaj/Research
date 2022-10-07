from utils3 import *
from numpy.linalg import *
from numpy import *
import numpy
import matplotlib.pyplot 

def Setup(N):
    np.random.seed(100)   
    d, a = Store(N)
    # d, a  = "./Deg.pkl", "./Adj.pkl"
    D, A = pkl.load(open(d,"rb")), pkl.load(open(a,"rb"))
    L = D - A
    A_gt = A

    np.random.seed(10)
    s0 = numpy.random.randn(N,)
    return L, A_gt, s0,D


def CreateNoisySamples(L,s0,dt,Lim,SNR): 
    N = L.shape[0]
    ewL, eVL = linalg.eig(L)
    SampleswoN = []
    Samples = []
    SampleswoN.append(s0)

    j = 0
    temp = ones(N,)

    while norm(SampleswoN[j]-temp) > Lim:
        j = j+1
        temp = dot(dot(dot(eVL, diag(exp(-ewL*dt*j))), eVL.T), s0)
        # Samples.append(SNR_adder(temp,SNR))
        SampleswoN.append(temp)
        temp = SampleswoN[j-1]
    del temp
    T = len(SampleswoN)
    TRAJ = s0.reshape(N,1)

    for i in range(1,T):
        TRAJ = np.append(TRAJ,SampleswoN[i].reshape(N,1),axis=1)

    Noisy_Traj = SNR_adder(TRAJ[0,:],SNR).reshape(1,T)

    for i in range(1,N):
        Noisy_Traj = np.append(Noisy_Traj, SNR_adder(TRAJ[i,:],SNR).reshape(1,T), axis =0)

    for i in range(T):
        Samples.append(Noisy_Traj[:,i])

    return Samples, T  

def Resample(k,Samples,dt):
    T = len(Samples)
    samples_k = [Samples[k*i] for i in range(T//k)]
    derivs_k = [(samples_k[i+1] - samples_k[i])/(k*dt) for i in range(T//k - 1)]
    return samples_k, derivs_k


def UVmatrix(T,samples_k, derivs_k,k,D):
    N = (samples_k[0]).shape[0]
    M = T//k -1 
    V = zeros((N,M))
    U = zeros((N,M))

    for ind in range(M):
        V[:,ind] = derivs_k[ind] + dot(D,samples_k[ind])
        U[:,ind] = samples_k[ind] 
    
    return U,V
    

def Learn(U,V):
    N = U.shape[0]
    M = U.shape[1]
    Bigu = U_synth(reshape(U[:,0], (N,1)))
    vV  = V[:,0]
    j = 1
    ranks = []

    for j in range(1,M):
        Uu = U_synth(reshape(U[:,j], (N,1)))
        Bigu = append(Bigu, Uu, axis=0)
        ranks.append(linalg.matrix_rank(Bigu))
        vV = append(vV, V[:,j], axis = 0)
    
    Adj_Learned = matricizer(dot(pinv(Bigu), vV))
    c = linalg.cond(Bigu)
    return Adj_Learned, c


def ErrCheck(Adj_le, A_gt):
    matplotlib.pyplot.matshow(A_gt)
    matplotlib.pyplot.matshow(Adj_le)
    return Adj_le, A_gt, norm(vectorizer(A_gt) - vectorizer(Adj_le))/(norm(vectorizer(A_gt)))


def CreateMulNoisySamp(L,s0,dt,Length,SNR):
    N = L.shape[0]
    ewL, eVL = linalg.eig(L)
    SampleswoN = []
    Samples = []
    SampleswoN.append(s0)

    j = 0
    temp = ones(N,)

    while j < Length:
        j = j+1
        temp = dot(dot(dot(eVL, diag(exp(-ewL*dt*j))), eVL.T), s0)
        # Samples.append(SNR_adder(temp,SNR))
        SampleswoN.append(temp)
        temp = SampleswoN[j-1]

    del temp 
    T = len(SampleswoN)
    TRAJ = s0.reshape(N,1)

    for i in range(1,T):
        TRAJ = np.append(TRAJ,SampleswoN[i].reshape(N,1),axis=1)

    Noisy_Traj = SNR_adder(TRAJ[0,:],SNR).reshape(1,T)

    for i in range(1,N):
        Noisy_Traj = np.append(Noisy_Traj, SNR_adder(TRAJ[i,:],SNR).reshape(1,T), axis =0)

    for i in range(T):
        Samples.append(Noisy_Traj[:,i])

    return Samples


def Combine(lengths,k,L,dt,SNR):
    N = L.shape[0]
    D = diag(diag(L))
    n = len(lengths)
    s = np.random.randn(N,n)
    sk,dk = Resample(k,CreateMulNoisySamp(L,s[:,0],dt,lengths[0],SNR),dt)
    U,V = UVmatrix(lengths[0]-1,sk,dk,k,D)

    for i in range(1,n):
        sk,dk = Resample(k,CreateMulNoisySamp(L,s[:,i],dt,lengths[i],SNR),dt)  
        b,c = UVmatrix(lengths[i]-1,sk,dk,k,D)
        U = append(U,b,axis=1)
        V = append(V,c,axis=1)
    return U,V

# def Combine(lengths,L,dt,SNR):
#     N = L.shape[0]
#     Samples = []
#     k = len(lengths)
#     s = np.random.randn(N,k)
#     for i in range(k):
#         Samples = Samples + CreateMulNoisySamp(L,s[:,i],dt,lengths[i],SNR)
#     return Samples







