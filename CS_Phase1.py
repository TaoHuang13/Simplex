import numpy as np
import CS_Phase2

# 构建初始单纯形表
def BuildAuxiliaryAbr(A, b, Cost):
    m, n = A.shape
    for i in range(m):
        if b[i] < 0:
            A[i] = A[i] * (-1)
            b[i] = -b[i]
    print(A)
    b_ = b.reshape(m,1)
    Identity = np.eye(m)
    AuxiliaryA = np.concatenate((A, Identity), axis=1)  # 构建A
    AuxiliaryAb = np.concatenate((AuxiliaryA, b_), axis=1)  # 构建Ab
    AuxiliaryCost = np.zeros(m+n+1)
    AuxiliaryCost[n:n+m] = 1
    AuxAbr = np.concatenate((AuxiliaryAb, AuxiliaryCost.reshape(1,m+n+1)), axis=0)  # 构建Abr
    M, N = AuxAbr.shape
    IndexB = np.arange(N-M, N-1)
    AuxAbr = InitialReducedCost(AuxAbr, IndexB) # 构建初始表格
    return AuxAbr


# 初始化检验数
def InitialReducedCost(AuxAbr, IndexB, Auxiliary = True):
    m_, n_ = AuxAbr.shape
    if Auxiliary == True:   # 对于Auxiliary问题的初始化检验数
        for i in range(m_ - 1):
            AuxAbr[m_-1,:] -= AuxAbr[i,:]
        return AuxAbr
    else:                   # 对于原问题的初始化检验数
        for i in range(len(IndexB)):
            AuxAbr[m_-1,:] -= AuxAbr[i,:] * AuxAbr[m_-1, IndexB[i]]
        return AuxAbr
    
# 第一阶段solver
def AuxSolver(A, b, r):
    AuxAbr = BuildAuxiliaryAbr(A, b, r)
    M, N = AuxAbr.shape
    IndexB = np.arange(N-M, N-1)
    Ab, IndexB, count = CS_Phase2.WholeIteration(AuxAbr, IndexB, Auxiliary=True)
    m, n = Ab.shape[0], Ab.shape[1]-1
    return Ab, IndexB, count
