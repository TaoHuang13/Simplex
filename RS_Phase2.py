import numpy as np
import copy
import random

# 0. 初始步骤
def InitialStep(IndexB, A, b, Cost):
    B = A[:,IndexB]
    B_inv = np.linalg.inv(B)
    b_bar = B_inv.dot(b)
    lamb = np.matmul(Cost[IndexB].T, B_inv).T
     
    return B_inv, b_bar, lamb

# 1. 用单纯形乘子计算检验数
def ComputeReducedCost(IndexN, A, Cost, lamb):
    c_N = Cost[IndexN]
    N = A[:, IndexN]
    r_N = (c_N.T - np.matmul(lamb.T, N)).T
    if np.min(r_N) >= -1e-6:
        return r_N, True
    else:
        return r_N, False

# 2. 选取进基变量
def InBasis(IndexN, r_N):
    assert r_N is not None  # If not, we should stop at previous step
    RealIndex = np.argsort(IndexN)
    for i in range(len(r_N)):
        if r_N[RealIndex[i]] < -1e-6:
            IndexOfN = RealIndex[i]
            break

    return IndexOfN

# 3. 选取出基变量
def OutBasis(A, B_inv, b_bar, IndexOfN, IndexN, IndexB):
    q = IndexN[IndexOfN]
    y_q = B_inv.dot(A[:,q])
    y_q.reshape(y_q.shape[0])
    if np.max(y_q) <= 0:    # Unbounded
        raise Exception('该问题无界')
    else:                   # Bounded
        y_q_posi = np.where(y_q > 0)
        ratio = b_bar[y_q_posi] / y_q[y_q_posi]
        IndexOfPosi = np.argmin(ratio)
        IndexOfB = y_q_posi[0][IndexOfPosi]

    return IndexOfB, y_q
    
# 4. 更新参数
def UpdateParams(r_N, B_inv, b_bar, lamb, IndexOfB, IndexOfN, y_q, IndexB, IndexN):
    size, _ = B_inv.shape   
    E = np.eye(size)
    p, q = IndexB[IndexOfB], IndexN[IndexOfN]
    E[:,IndexOfB] = - y_q / y_q[IndexOfB]
    E[IndexOfB,IndexOfB] = 1 / y_q[IndexOfB] 
    # Update
    New_B_inv = np.matmul(E, B_inv)
    New_b_bar = np.matmul(E, b_bar)
    New_lamb = lamb + (r_N[IndexOfN] / y_q[IndexOfB] * B_inv[IndexOfB,:]).T     
    IndexB[np.where(IndexB==p)] = q
    IndexN[np.where(IndexN==q)] = p

    return New_B_inv, New_b_bar, New_lamb, IndexB, IndexN

# 完整迭代
def Phase2(A, b, Cost, IndexB, IndexN):
    New_B_inv, New_b_bar, New_lamb = InitialStep(IndexB, A, b, Cost)
    NewIndexB, NewIndexN = copy.deepcopy(IndexB), copy.deepcopy(IndexN)
    count = 0
    print('Phase2: Iteration ' + str(count))
    print('Basis: ', NewIndexB)
    print('Multiplier: ', New_lamb)
    print('---------------------------')
    # 如果是线性系统的话
    if len(IndexN) == 0:
        if np.min(New_b_bar) < 0:
            raise Exception('原问题无可行解，这边涉及到精度问题，所以如果与答案不符，请助教调整下代码中的精度')
        return NewIndexB, New_b_bar, count

    while True:
        count += 1
        r_N, stop = ComputeReducedCost(NewIndexN, A, Cost, New_lamb)    # Step1
        if stop:
            if np.min(New_b_bar) < 0:
                raise Exception('原问题无可行解，这边涉及到精度问题，所以如果与答案不符，请助教调整下代码中的精度')
            return NewIndexB, New_b_bar, count
        else:
            IndexOfN = InBasis(NewIndexN, r_N)  # Step2
            IndexOfB, y_q = OutBasis(A, New_B_inv, New_b_bar, IndexOfN, NewIndexN, NewIndexB)   # Step 3
            New_B_inv, New_b_bar, New_lamb, NewIndexB, NewIndexN = \
                UpdateParams(r_N, New_B_inv, New_b_bar, New_lamb, IndexOfB, \
                    IndexOfN, y_q, NewIndexB, NewIndexN) # Step 4
        print('Phase2: Iteration ' + str(count))
        print('Basis: ', NewIndexB)
        print('Multiplier: ', New_lamb)
        print('---------------------------')

# 辅助变量
def Auxiliary(A, b, Cost, IndexB, IndexN):
    n, m_n = A.shape
    m = m_n - n # keep A's shape
    New_B_inv, New_b_bar, New_lamb = InitialStep(IndexB, A, b, Cost)
    NewIndexB, NewIndexN = copy.deepcopy(IndexB), copy.deepcopy(IndexN)
    count = 0
    print('Phase1: Iteration ' + str(count))
    print('Basis: ', NewIndexB)
    print('Multiplier: ', New_lamb)
    print('---------------------------')
    while True:
        count += 1
        r_N, stop = ComputeReducedCost(NewIndexN, A, Cost, New_lamb)    # Step1
        if stop:
            A, b, NewIndexB, NewIndexN, New_b_bar = CheckAuxiliary(A, b, Cost, NewIndexB, NewIndexN, New_B_inv, New_b_bar, New_lamb, count, m)
            return A, b, NewIndexB, NewIndexN, New_b_bar, count
        else:
            IndexOfN = InBasis(NewIndexN, r_N)  # Step2
            IndexOfB, y_q = OutBasis(A, New_B_inv, New_b_bar, IndexOfN, NewIndexN, NewIndexB)   # Step 3
            New_B_inv, New_b_bar, New_lamb, NewIndexB, NewIndexN = UpdateParams(r_N, New_B_inv, New_b_bar, New_lamb, IndexOfB, IndexOfN, y_q, NewIndexB, NewIndexN) # Step 4
        print('Phase1: Iteration ' + str(count))
        print('Basis: ', NewIndexB)
        print('Multiplier: ', New_lamb)
        print('---------------------------')

        AuxSolution = np.zeros(m_n)
        for i in range(len(NewIndexB)):
            AuxSolution[NewIndexB[i]] = New_b_bar[i]
        AuxValue = AuxSolution.dot(Cost)

    if AuxValue > 1e-5 or np.min(AuxSolution) < 0:
        raise Exception('原问题无可行解，这边涉及到精度问题，所以如果与答案不符，请助教调整下代码中的精度')
    return NewIndexB, b_bar, count

# 检查人工变量
def CheckAuxiliary(A, b, Cost, NewIndexB, NewIndexN, New_B_inv, New_b_bar, New_lamb, count, m):
    n = len(NewIndexB)
    AuxVariablesIndex = np.where(NewIndexB >= m)
    count_ = count
    while len(AuxVariablesIndex[0]) != 0: # 存在人工变量
        OutBasisIndex = random.choice(AuxVariablesIndex)[0]
        A, b, NewIndexB, NewIndexN, New_B_inv, New_b_bar, New_lamb,InBasisIndex, y_q = AuxiliaryInBasis(A, b, Cost, NewIndexB, NewIndexN, New_B_inv, New_b_bar, New_lamb, OutBasisIndex, count_, m)
        if y_q is None: # 冗余约束
            AuxVariablesIndex = np.where(NewIndexB >= m)
            continue
        else:   # 手动剔除人工变量
            r_N, stop = ComputeReducedCost(NewIndexN, A, Cost, New_lamb)
            New_B_inv, New_b_bar, New_lamb, NewIndexB, NewIndexN = UpdateParams(r_N, New_B_inv, New_b_bar, New_lamb, OutBasisIndex, InBasisIndex, y_q, NewIndexB, NewIndexN)
            AuxVariablesIndex = np.where(NewIndexB >= m)
            print('Phase1: Iteration ' + str(count_))
            print('Basis: ', NewIndexB)
            print('Multiplier: ', New_lamb)
            print('---------------------------')
            
        count_ += 1
    return A, b, NewIndexB, NewIndexN, New_b_bar

# 确定进基非人工变量
def AuxiliaryInBasis(A, b, Cost, NewIndexB, NewIndexN, New_B_inv, New_b_bar, New_lamb, OutBasisIndex, count_, m):
    n = New_B_inv.shape[0]
    Y = New_B_inv.dot(A[:,NewIndexN])
    # 判断是否为冗余约束
    if np.mean(np.abs(Y[OutBasisIndex, NewIndexN < m])) <= 1e-5:    # 均为0
        A_ = np.delete(A, OutBasisIndex, axis=0)
        b_ = np.delete(b, OutBasisIndex)
        NewIndexB_ = np.delete(NewIndexB, OutBasisIndex)
        NewIndexN_ = np.append(NewIndexN, NewIndexB[OutBasisIndex])
        New_B_inv_ =  np.linalg.inv(A_[:,NewIndexB_])
        New_b_bar_ = New_B_inv_.dot(b_)
        New_lamb_ = np.matmul(Cost[NewIndexB_].T, New_B_inv_).T
        Y = New_B_inv_.dot(A_[:,NewIndexN])
        InBasisIndex, y_q = None, None
        print('Phase1 with redundant constrain: Iteration ' + str(count_))
        print('Basis: ', NewIndexB_)
        print('Multiplier: ', New_lamb_)
        print('---------------------------')
        return A_, b_, NewIndexB_, NewIndexN_, New_B_inv_, New_b_bar_, New_lamb_, InBasisIndex, y_q

    InBasisIndex = -1
    for i in range(len(NewIndexN)):
        InBasisIndex = 1000000
        if Y[OutBasisIndex,i] !=0 and NewIndexN[i] < m:
            InBasisIndex = i
            break
    if InBasisIndex == 1000000:
        raise Exception('这种情况我没有测试过，如果代码没有问题的话，抛出异常表示无法解辅助问题')
    y_q = Y[:,InBasisIndex]
    return A, b, NewIndexB, NewIndexN, New_B_inv, New_b_bar, New_lamb, InBasisIndex, y_q

