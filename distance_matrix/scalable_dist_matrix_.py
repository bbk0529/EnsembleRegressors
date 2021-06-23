import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import random
import sys


#Data Preparation
class DistanceMatrix() : 
    # N = number of stations 
    # s = number of desired slicing
    # p = paramter for the padding 

    # def __init__ (self, data: np.ndarray, s: int = 100, p: float = 1) : 
    #     #loading the data 
    #     self._data = data               
    #     self._setup_parameters(s,p)

    def __init__ (self, N: int = 10000, s: int = 100, p: float = 1) : 
        # random data generated 
        self._data = self.__randomdata_generator(N)       
        self._setup_parameters(s,p)        

    def __randomdata_generator(self, N) : 
        return np.random.rand(N,2)
        
    def _setup_parameters(self,s,p) : 
        self._idx = np.arange(0, len(self._data), 1)        
        self._DIST = {}
        self._IDX_w_padding = {}        
        self._IDX_masking = {}       
        self.NEIGHBOR={}

        # reading the max and min 
        x_min = np.min(self._data[:,0])
        x_max = np.max(self._data[:,0])
        y_min = np.min(self._data[:,1])
        y_max = np.max(self._data[:,1])
        
        # defining interval 
        x_interval = (x_max - x_min) / s 
        y_interval = (y_max - y_min) / s        

        self._x_pad = x_interval * p
        self._y_pad = y_interval * p
        
        self._lst_x = np.arange(x_min, x_max + x_interval, x_interval)
        self._lst_y = np.arange(y_min, y_max + y_interval, y_interval)    




    #naive approach
    def dist_matrix(self) :         
        self.DIST = distance_matrix(self._data, self._data)
    
    #Proposed algorithm
    def create_dist_matrix(self) : 
        for i in range(len(self._lst_x)-1) : 
            x_idx_w_padding = (self._lst_x[i] - self._x_pad <= self._data[:,0]) & (self._data[:,0]<= self._lst_x[i+1] + self._x_pad) #boolean array             
            for j in range(len(self._lst_y)-1) : 
                y_idx_w_padding = (self._lst_y[j] - self._y_pad <= self._data[:,1]) & (self._data[:,1]<= self._lst_y[j+1] + self._y_pad)           
                
                # list of points in the area including padding 
                idx_clip_padding = self._idx[x_idx_w_padding & y_idx_w_padding]
                
                #distance matrix for clipped data
                data_clip = self._data[idx_clip_padding]
                dist_matrix_clip = distance_matrix(data_clip, data_clip)    
                
                #masking array to filter out center points 
                idx_clip_boolean_x = (self._lst_x[i] <= data_clip[:,0]) & (data_clip[:,0] <=self._lst_x[i+1])
                idx_clip_boolean_y = (self._lst_y[j] <= data_clip[:,1]) & (data_clip[:,1] <=self._lst_y[j+1])
                idx_clip_boolean = idx_clip_boolean_x & idx_clip_boolean_y
                
                #store the computation in the dictionary 
                self._DIST[i,j] = dist_matrix_clip        
                self._IDX_w_padding[i,j] = idx_clip_padding                
                self._IDX_masking[i,j] = idx_clip_boolean
    
    def find_neighbor(self, k: int=11) :        
        for key in self._DIST.keys()  : # keys (i,j)
            try :                             
                block_dist_mat_argsort = self._DIST[key].argsort() #list of index number (arg) in increasing order, based on the distance 
                block_idx = self._IDX_w_padding[key]    
                block_idx_sorted = block_idx[block_dist_mat_argsort]            
                block_idx_sorted_filtered = block_idx_sorted[self._IDX_masking[key]]
                
                for i in block_idx_sorted_filtered : 
                    center_station = i[0]
                    neighbors = i[1:k+1]
                    self.NEIGHBOR[center_station] = neighbors
            except :
                print(key, "no entity") 
                pass

    #Test Purpose
    def pick_and_draw_random_block(self) :                      
        while True: 
            random_block_idx = random.choice(list(self._IDX_w_padding.keys()))
            points = self._IDX_w_padding[random_block_idx]
            mask = self._IDX_masking[random_block_idx]

            if len(points[mask]) > 0 : 
                break
    
        
        padding_points = self._data[points]
        center_points = self._data[points[mask]]
        target_point = random.choice(points[mask])

        fig,axes = plt.subplots(1,2,figsize=(20,10))
        
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)

        self.draw(ax1, self._data, self.NEIGHBOR, random_block_idx, target_point, center_points, padding_points, False)
        self.draw(ax2, self._data, self.NEIGHBOR, random_block_idx, target_point, center_points, padding_points, True)
        
        plt.show()

    
    def draw(self, axe, data, neighbor, random_block_idx, target_point,  center_points, padding_points, zoom) : 
            x = 0 
            y = 1


            # axe.plt.figure(figsize=(10,10))
            if zoom : 
                lst = []
                for i in [-1,0,1] : 
                    for j in [-1,0,1] : 
                        try : 
                            lst.extend(self._IDX_w_padding[random_block_idx[x]+i, random_block_idx[y]+j])
                        except :
                            pass                            
                axe.scatter(data[lst][:,x], data[lst][:,y], color = 'lightgrey', s=10)
                size = [100,200]
            else : 
                axe.scatter(data[:,x], data[:,y], color = 'lightgrey', s=10)
                size = [10,20]

            
            
            
            
            # axe.title(str(random_block_idx) + str(target_point) )        
            axe.scatter(padding_points[:,x], padding_points[:,y], color='darkgrey', s=size[0])        
            axe.scatter(center_points[:,x], center_points[:,y], color='green', s=size[0])
            
            axe.scatter(data[neighbor[target_point]][:,x], data[neighbor[target_point]][:,y], color='blue', s=size[1])
            axe.scatter(data[target_point][x], data[target_point][y], color='red', s=size[1])
            axe.legend([
                'points out of boundary',
                'points within padding',
                'center points',
                'neighbors',
                'target point'
                        
                ]
            )
            

def test(n,s,p) : 
    print("\n"*2)
    print("="*100)
    print(n,s,p)
    distmat = DistanceMatrix(n,s,p) 
    print("{} stations \n slicing by {}\n with padding {}".format(n, s, p))
    start = time.time()
    distmat.create_dist_matrix()
    distmat.find_neighbor(k=10)    
    end = time.time() - start    
        

    matrix_size = 0
    for key in distmat._DIST.keys() : 
        matrix_size += distmat._DIST[key].shape[0] * distmat._DIST[key].shape[1]
    
    matrix_original_size = len(distmat._data)**2
    factor = round(len(distmat._data)**2 / matrix_size,2)

    filename = 'test_result.txt'
    f = open(filename, 'a')
    result = [n,s,p,round(end,4), matrix_original_size, matrix_size, factor]    
    f.write("\n")
    f.write(str(result))
    f.close()
    
    print("="*100)    
    print("\t elapsted time {}".format(round(end,3)))
    print("\t size of NEIGHBOR:", len(distmat.NEIGHBOR))
    print("\t size of dist_matrix with the original: {:,}".format(matrix_original_size))
    print("\t size of dist_matrix with the reduced: {:,}".format(matrix_size))
    print("\t improved by the factor of: {}".format(factor))
    print("="*100)    


if __name__ == '__main__' : 
    # N = int(sys.argv[1])  
    # s = int(sys.argv[2])  
    # p = float(sys.argv[3])
    # k = int(sys.argv[4])
    import signal
    # N = [3200, 6400, 12800,25600,50000, 100000, 200000]
    N = [10000]
    S = [50, 100, 200, 300, 400]
    P = [0.5,1,1.5, 2]
    # K = [1,2,4,8,10,20,40,50,70, 100]

    

    def handler(signum, frame):
        print("timeout")
        raise Exception("end of time")

    for n in N : 
        for s in S : 
            for p in P : 
                if n <= s : 
                    continue      
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(300)
                try : 
                    test(n,s,p)
                except Exception as e:
                    print(e)


    # print(DIST)
    


    # distmat.pick_and_draw_random_block()
    


    
    #     print(metadata._data.shape)
    #     DIST = distance_matrix(metadata._data, metadata._data)
    #     print(DIST.shape)
    #     end = time.time() - start
    #     print("\n", i, end)
    #     result.append((i, end))
    
    # dist_matrix_creator


    