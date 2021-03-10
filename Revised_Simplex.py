import numpy as np
import RS_Phase1
import RS_Phase2
import time

def RevisedSimplex(A, b, Cost):
    start = time.time()
    n, m = A.shape
    A, b, NewIndexB, NewIndexN, New_b_bar, count1 = RS_Phase1.AuxiliarySolver(A, b, Cost)
    A = A[:m]
    NewIndexN = NewIndexN[NewIndexN < m]
    print('Phase1: Output')
    print('Basis: ', NewIndexB)
    print('---------------------------')
    NewIndexB, b_bar, count2 = RS_Phase2.Phase2(A, b, Cost, NewIndexB, NewIndexN) 
    OptimalSolution = np.zeros(m)
    for i in range(len(NewIndexB)):
        OptimalSolution[NewIndexB[i]] = b_bar[i]
    end = time.time()
    print('We are done!')
    print('Basis: ', NewIndexB)
    print('Optimal Solution: ', OptimalSolution)
    print('Optimal Value: %.2f ' % OptimalSolution.dot(Cost))
    print('Number of Pivoting: ', str(count1+count2))
    print('Total Running Times: ', str(end-start))