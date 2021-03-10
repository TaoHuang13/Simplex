import numpy as np
import copy
import random
import CS_Phase1
precise = 1e-6

# 检验是否终止
def CheckTerminal(r):
    return True if np.min(r) >= -precise else False

# 确定进基变量
def InBasis(r):
    for i in range(len(r)):
        if r[i] < -precise:
            q = i
            break
    return q

# 确定出基变量
def OutBasis(Y, b_bar, q):
    Y_q = Y[:,q]
    print(Y_q)
    if np.max(Y_q) <= precise:    # 无界
        raise Exception('该问题无界')
    else:
        Y_q_posi = np.where(Y_q > precise)
        ratio = b_bar[Y_q_posi] / Y_q[Y_q_posi]
        IndexOfPosi = np.argmin(ratio)
        p = Y_q_posi[0][IndexOfPosi]
        return p

# 更新表格
def UpdateTable(Abr, p, q):
    m, n = Abr.shape
    for i in range(m):
        if i != p:
            Abr[i,:] = Abr[i,:] - Abr[i,q] / Abr[p,q] * Abr[p,:]
        else:
            Abr[i,:] = Abr[i,:] / Abr[i,q]

# 完整迭代过程
def WholeIteration(Abr, IndexB, Auxiliary = True):
    M, N = Abr.shape
    n = N - M
    Y, b_bar, r = Abr[0:M-1, 0:N-1], Abr[0:M-1, N-1], Abr[M-1, 0:N-1]
    count = 0
    print('Phase1: Iteration ' + str(count)) if Auxiliary else print('Phase2: Iteration ' + str(count))
    print(Abr)
    print('Basis: ', IndexB)
    print('---------------------------')
    while CheckTerminal(r) == False:
        count += 1
        print('Phase1: Iteration ' + str(count)) if Auxiliary else print('Phase2: Iteration ' + str(count))
        q = InBasis(r)
        p = OutBasis(Y, b_bar, q)
        IndexB[p] = q
        UpdateTable(Abr, p, q)
        Y, b_bar, r = Abr[0:M-1, 0:N-1], Abr[0:M-1, N-1], Abr[M-1, 0:N-1]
        print(Abr)
        print('Basis: ', IndexB)
        print('---------------------------')

    # 对于辅助问题需要检验人工变量是否清除
    if Auxiliary == True:
        AuxVariablesIndex = np.where(IndexB >= n)
        while len(AuxVariablesIndex[0]) > 0:    #如果存在冗余约束
            count += 1
            OutBasisIndex = random.choice(AuxVariablesIndex)[0]
            if np.mean(np.abs(Y[OutBasisIndex, 0:n])) <= 1e-6:   # 删除一行
                print('Phase1 with redundant constrain: Iteration ' + str(count))
                Abr = np.delete(Abr, OutBasisIndex, axis=0)
                b_bar = np.delete(b_bar, OutBasisIndex)
                M = M - 1
                IndexB = np.delete(IndexB, OutBasisIndex)
            else:                                            # 去掉一个人工变量
                q = 1000000
                for i in range(n):
                    if Abr[OutBasisIndex, i] != 0 and i not in IndexB:
                        q = i
                        break
                if q == 1000000:
                    raise Exception('这种情况我没有测试过，如果代码没有问题的话，抛出异常表示无法解辅助问题')
                IndexB[OutBasisIndex] = q
                UpdateTable(Abr, OutBasisIndex, q)
                Y, b_bar, r = Abr[0:M-1, 0:N-1], Abr[0:M-1, N-1], Abr[M-1, 0:N-1]
            print('Phase1: Iteration ' + str(count))
            print(Abr)
            print('Basis: ', IndexB)
            print('---------------------------')
            Y, b_bar, r = Abr[0:M-1, 0:N-1], Abr[0:M-1, N-1], Abr[M-1, 0:N-1]
            AuxVariablesIndex = np.where(IndexB > n)
            
        if abs(Abr[-1,-1]) >= 1e-5 or min(b_bar) < 0:
            raise Exception("原问题无可行解，这边涉及到精度问题，所以如果与答案不符，请助教调整下我这边的精度")
        Ab = np.concatenate((Y[:,0:n],b_bar.reshape(M-1,1)), axis=1)
        return Ab, IndexB, count
    else:
        Ab = np.concatenate((Y,b_bar.reshape(M-1,1)), axis=1)
        return Ab, IndexB, count

# 第二阶段solver
def NormalSolver(Ab, Cost, IndexB):
    Cost_ = np.concatenate((Cost, [0])).reshape(1,-1)
    Abr = np.concatenate((Ab, Cost_), axis=0)
    Initial_Abr = CS_Phase1.InitialReducedCost(Abr, IndexB, Auxiliary=False)
    Ab, IndexB, count = WholeIteration(Abr, IndexB, Auxiliary=False)
    m, n = Ab.shape
    return Ab[:,n-1], IndexB, count
    

