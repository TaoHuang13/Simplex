import numpy as np
import RS_Phase2

# 构建单纯形表
def BuildTable(A, b, Cost):
    n, m = A.shape
    for i in range(n):
        if b[i] < 0:
            A[i] = A[i] * -1
            b[i] = -b[i]
    Identity = np.eye(n)
    NewA = np.concatenate((A, Identity), axis=1)    # 新的表格
    NewCost = np.zeros(m+n).T
    NewCost[m:] = 1
    IndexB = np.arange(m, m+n)
    IndexN = np.arange(0,m)

    return NewA, NewCost, IndexB, IndexN

# 解辅助问题
def AuxiliarySolver(A, b, Cost):
    NewA, NewCost, IndexB, IndexN = BuildTable(A, b, Cost)
    A, b, NewIndexB, NewIndexN, New_b_bar, count = RS_Phase2.Auxiliary(NewA, b, NewCost, IndexB, IndexN)

    return A, b, NewIndexB, NewIndexN, New_b_bar, count