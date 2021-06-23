import numpy as np
from scipy.spatial import distance_matrix
import sys
import random


class Dist_Matrix() : 
    
    Neighbor = {}
    
    def __init__(self, N):
        self.data = np.random.rand(N,2)
        self.idx = np.arange(0, len(self.data), 1)        

    def find_neighbor(self, data,r) : 
        if len(data) == 0 : 
            return 
        min_x = np.min(data[:,0])
        max_x = np.max(data[:,0])
        min_y = np.min(data[:,1])
        max_y = np.max(data[:,1])
        
        
        idx_padding_x = (min_x - r  <= self.data[:,0]) & (self.data[:,0] <= max_x + r)
        idx_padding_y = (min_y - r <= self.data[:,1]) & (self.data[:,1] <= max_y + r)
        idx_padding = idx_padding_x & idx_padding_y
        idx_clip_padding = self.idx[idx_padding]

        clipped_data = self.data[idx_padding]

        masking_idx_x = (min_x  <= clipped_data[:,0]) & (clipped_data[:,0] <= max_x)
        masking_idx_y = (min_y  <= clipped_data[:,1]) & (clipped_data[:,1] <= max_y)
        
        masking_idx = masking_idx_x & masking_idx_y
      
  
        dist_matrix = distance_matrix(self.data[idx_clip_padding], self.data[idx_clip_padding])
        dist_matrix_arg_idx = dist_matrix.argsort()
        result = idx_clip_padding[dist_matrix_arg_idx][masking_idx]
        
        for r in result : 
            self.Neighbor[r[0]] = r[1:]      


    def splitting_data(self, data, n, r) : 
        if len(data) > 4 * n  :
            min_x = np.min(data[:,0])
            max_x = np.max(data[:,0])
            min_y = np.min(data[:,1])
            max_y = np.max(data[:,1])
            mid_x = (max_x + min_x) / 2
            mid_y = (max_y + min_y) / 2

            # mid_x = random.choice(data[:,0])
            # mid_y = random.choice(data[:,1])


            idx_1 = (data[:,0] <= mid_x) & (data[:,1] <= mid_y)
            idx_2 = (data[:,0] <= mid_x) & (data[:,1] >= mid_y)
            idx_3 = (data[:,0] >= mid_x) & (data[:,1] <= mid_y)
            idx_4 = (data[:,0] >= mid_x) & (data[:,1] >= mid_y)        

            data_1 = data[idx_1]
            data_2 = data[idx_2]
            data_3 = data[idx_3]
            data_4 = data[idx_4]
            DATA = [data_1, data_2, data_3, data_4]        
            

            for d in DATA :                             
                self.splitting_data(d, n, r)
        else : 
            self.find_neighbor(data, r)                


    # def splitting_data(self, data, n, r) : 
    #     min_x = np.min(data[:,0])
    #     max_x = np.max(data[:,0])
    #     min_y = np.min(data[:,1])
    #     max_y = np.max(data[:,1])
    #     mid_x = (max_x + min_x) / 2
    #     mid_y = (max_y + min_y) / 2
    #     idx_1 = (data[:,0] <= mid_x) & (data[:,1] <= mid_y)
    #     idx_2 = (data[:,0] <= mid_x) & (data[:,1] >= mid_y)
    #     idx_3 = (data[:,0] >= mid_x) & (data[:,1] <= mid_y)
    #     idx_4 = (data[:,0] >= mid_x) & (data[:,1] >= mid_y)        

    #     data_1 = data[idx_1]
    #     data_2 = data[idx_2]
    #     data_3 = data[idx_3]
    #     data_4 = data[idx_4]
    #     DATA = [data_1, data_2, data_3, data_4]        
        

    #     for d in DATA :             
    #         if (1 <= len(d)) & (len(d) <= n) : 
    #             self.find_neighbor(d, r)                
    #         elif len(d) > n :             
    #             self.splitting_data(d, n, r)


if __name__ == '__main__' : 
    N = int(sys.argv[1])
    n = int(sys.argv[2])
    r = float(sys.argv[3])
    # N = 20000
    # n = 10
    # r = 0.01

    # print(data)
    dm = Dist_Matrix(N)
    import time
    start = time.time()
    r = 0.01
    dm.splitting_data(dm.data, n, r)    
    end = time.time()
    print(len(dm.Neighbor))
    print(end - start)


