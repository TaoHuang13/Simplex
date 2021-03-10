import numpy as np
import CS_Phase1
import CS_Phase2
import time

def ClassicalSimplex(A, b, Cost):
    start = time.time()
    np.set_printoptions(precision=2, suppress=True)
    m, n = A.shape
    Ab, IndexB, count1 = CS_Phase1.AuxSolver(A, b, Cost)
    print('Phase1: Output A_b without r')
    print(Ab)
    print('Basis: ', IndexB)
    print('---------------------------')
    b_bar, IndexB, count2 = CS_Phase2.NormalSolver(Ab, Cost, IndexB)
    OptimalSolution = np.zeros(n)
    for i in range(len(IndexB)):
        OptimalSolution[IndexB[i]] = b_bar[i]
    end = time.time()
    print('We are done!')
    print('Basis: ', IndexB)
    print('Optimal Solution: ', OptimalSolution)
    print('Optimal Value: %.2f ' % OptimalSolution.dot(Cost))
    print('Number of Pivoting: ', str(count1+count2))
    print('Total Running Times: ', str(end-start))