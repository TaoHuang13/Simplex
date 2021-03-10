import numpy as np
import time
import Classical_Simplex
import Revised_Simplex

# Testcase1
# A = np.loadtxt('./data1/A.csv', delimiter=',')
# b = np.loadtxt('./data1/b.csv', delimiter=',')
# Cost = np.loadtxt('./data1/c.csv', delimiter=',')

# Testcase2
# A = np.array([[6, 1, -2, -1, 0, 0],[1, 1, 1, 0, 1, 0],[6, 4, -2, 0, 0, -1]])
# b = np.array([5,4,10]).T
# Cost = np.array([5,2,-4,0,0,0]).T

# Testcase3
# A = np.array([[1,0,0,1,0,0],[20,1,0,0,1,0],[200,20,1,0,0,1]])
# b = np.array([1,100,10000]).T
# Cost = np.array([-100,-10,-1,0,0,0]).T

# Testcase4 redundant constraint
A = np.array([[2,-1,1,0], [1,1,1,0], [1,0,0,1],[3,0,2,0]])
b = np.array([4,6,2,10]).T
Cost = np.array([-3,-1,2,0]).T

# Testcase5 Unbounded
# A = np.array([[1,0]])
# b = np.array([0]).T
# Cost = np.array([-1,-1]).T

# Testcase6 negative b 
# A = np.array([[-6, -1, 2, 1, 0, 0],[-1, -1, -1, 0, -1, 0],[6, 4, -2, 0, 0, -1]])
# b = np.array([-5,-4,10]).T
# Cost = np.array([5,2,-4,0,0,0]).T

# A = np.array([[2,1],[1, 1]])
# b = np.array([-3,-4]).T
# Cost = np.array([-1,-1]).T

# Testcase 7 degenerate
# A = np.array([[1,0,0,1/4,-8,-1,9],[0,1,0,1/2,-12,-1/2,3],[0,0,1,0,0,1,0]])
# b = np.array([0,0,1]).T
# Cost = np.array([0,0,0,-3/4,20,-1/2,6]).T

# A = np.array([[0.25,-60,-1/25,9,1,0,0], [1/2,-90,-1/50,3,0,1,0],[0,0,1,0,0,0,1]])
# b = np.array([0,0,1]).T
# Cost = np.array([-3/4,150,-1/50,6,0,0,0]).T

# A = np.array([[1,-1,0,1,0,0],[2,0,1,0,1,0],[1,1,1,0,0,1]])
# b = np.array([2,4,3]).T
# Cost = np.array([2,0,1.5,0,0,0]).T


def main(A, b, cost, method = 'classical'):
    if method == 'classical':
        Classical_Simplex.ClassicalSimplex(A, b, cost)
    elif method == 'revised':
        Revised_Simplex.RevisedSimplex(A, b, cost)


main(A, b, Cost, method='classical') 

