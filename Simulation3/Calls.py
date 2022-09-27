from Heat_atom import *

L, A_gt, s0,D = Setup(4)
Samples, T = CreateNoisySamples(L,s0,0.001,1e-5,10)
sk, dk = Resample(3,Samples,0.001)
U,V = UVmatrix(T,sk,dk,3,D)
Adj, Cond = Learn(U,V)
ErrCheck(Adj, A_gt)