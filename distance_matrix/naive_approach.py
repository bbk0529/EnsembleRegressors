from scipy.spatial import distance_matrix
import numpy as np
import time
import sys


def test(n) : 
    print("{} stations \n".format(n))
    dist_matrix = np.random.rand(n, 2)
    start = time.time()
    DIST = distance_matrix(dist_matrix, dist_matrix)    
    duration = time.time() - start       
    
    filename = 'test_result_naive_2.txt'    
    f = open(filename, 'a')
    result = [n, duration]    
    f.write("\n")
    f.write(str(result)[1:-1])
    f.close()

    print(duration)
    print("="*100)    

if __name__ == '__main__' : 
    N = int(sys.argv[1])  
    test(N)