import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import random
import sys
import math 

#Data Preparation
class DistanceMatrix() : 
    # N = number of stations 
    # s = number of desired slicing
    # p = paramter for the padding 

    def __init__(self, data: np.ndarray, n: int, k: int) : 
        self.data = data
        self.N = len(self.data)
        self._idx = np.arange(0, self.N, 1)        
        self._DIST = {}
        self._IDX_w_padding = {}        
        self._IDX_masking = {}               
        self._no_of_points_in_unit = n
        self._no_of_neighbors = k

        self.NEIGHBOR = {}
        self.stat = {}
        self.flag = {} 
        self.flag['create_distance_matrix'] = False 
        self.flag['find_neighbor'] = False 
        self.flag['flag_compute_stats'] = False
        # k = numbers of points to be possibly in the unit block if uniformly distributed        
        N = len(data)
        # n = number of points in unit
        # if k == None :
            # k = max(3, math.ceil(30 * 100000 / N))            
        
        self._setup_parameters()
        
    @classmethod
    def randomdata_generator(self, N) : 
        return np.random.rand(N,2)
        
    # k means "number of nearest stations "
    def _setup_parameters(self) : 
        # reading the max and min 
        self._x_min = np.min(self.data[:,0])
        self._x_max = np.max(self.data[:,0])
        self._y_min = np.min(self.data[:,1])
        self._y_max = np.max(self.data[:,1])

        

        self._lx = (self._x_max - self._x_min)
        self._ly = (self._y_max - self._y_min)
        total_area = self._lx * self._ly
        
        self._interval = math.sqrt(total_area * self._no_of_points_in_unit / self.N)        
        
        self._no_of_blocks_in_x = math.ceil(self._lx / self._interval)
        self._no_of_blocks_in_y = math.ceil(self._ly / self._interval)


        # minimum padding width = 1/2 * sqrt(k/N * lx * ly)        
        self._pad = 1/2 * math.sqrt(self._no_of_neighbors/self.N * self._lx * self._ly)
        
               
        # interval is setout     
        self._lst_x = np.arange(self._x_min, self._x_max + self._interval, self._interval)
        self._lst_y = np.arange(self._y_min, self._y_max + self._interval, self._interval)    
  


    # @property
    # def s(self) :
    #     return self._s
    
    # @s.setter
    # def s(self, val) :
    #     self._s = val


    # @property
    # def x_pad(self) :
    #     return self._x_pad
    
    # @x_pad.setter
    # def x_pad(self, val) :
    #     self._x_pad = val


    # def y_pad(self) :
    #     return self._y_pad
    
    # @y_pad.setter
    # def y_pad(self, val) :
    #     self._y_pad = val
        
    def create_dist_matrix(self) :
        start = time.time()        
        for i in range(self._no_of_blocks_in_x) : 
            x_idx_w_padding = (self._lst_x[i] - self._pad <= self.data[:,0]) & (self.data[:,0]<= self._lst_x[i+1] + self._pad) #boolean array             
            for j in range(self._no_of_blocks_in_y) : 
                y_idx_w_padding = (self._lst_y[j] - self._pad <= self.data[:,1]) & (self.data[:,1]<= self._lst_y[j+1] + self._pad)           
                
                # list of points in the area including padding 
                idx_clip_padding = self._idx[x_idx_w_padding & y_idx_w_padding]
                
                #distance matrix for clipped data
                data_clip = self.data[idx_clip_padding]
                dist_matrix_clip = distance_matrix(data_clip, data_clip)    
                
                #masking array to filter out center points 
                idx_clip_boolean_x = (self._lst_x[i] <= data_clip[:,0]) & (data_clip[:,0] <=self._lst_x[i+1])
                idx_clip_boolean_y = (self._lst_y[j] <= data_clip[:,1]) & (data_clip[:,1] <=self._lst_y[j+1])
                idx_clip_boolean = idx_clip_boolean_x & idx_clip_boolean_y
                
                #store the computation in the dictionary 
                self._DIST[i,j] = dist_matrix_clip        
                self._IDX_w_padding[i,j] = idx_clip_padding                
                self._IDX_masking[i,j] = idx_clip_boolean
        
        duration = time.time() - start
        self.stat['elapsed_time_create_dist_matrix'] = duration
        print("Completed Distance matrix with size of {0} made in {1} sec".format(len(self._DIST), round(duration,3)))
        self.flag['create_distance_matrix'] = True
    
    def find_neighbor(self, k: int=10) :        
        if not self.flag['create_distance_matrix'] :
            return "create_distance_matrix() has not successfully run yet"
        start = time.time()        
        for key in self._DIST.keys() : 
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
            
        duration = time.time() - start
        self.stat['elapsed_time_find_neighbor'] = duration
        print("Completed Neighbors dict with size of {0} made in {1} sec".format(len(self.NEIGHBOR), round(duration,3)))
        self.flag['find_neighbor'] = True


    def compute_stats(self) : 
        if not self.flag['find_neighbor'] :
            return "find_neighbor has not yet completed"

        unit_result = [] 
        for v in self._IDX_masking.values() : 
            unit_result.append(sum(v))
        
        pad_result = [] 
        for v in self._IDX_w_padding.values() : 
            pad_result.append(len(v))
        result = [] 
        for v in self.NEIGHBOR.values() : 
            result.append(len(v))
        import statistics        
        
        self.stat['area_of_unit']  = self._interval ** 2
        self.stat['expected_no_of_points_by_area_of_unit']  = self._interval ** 2 * self.N
        self.stat['area_of_unit_incl_padding']  = (self._interval + 2 * self._pad)** 2
        self.stat['expected_no_of_points_by_area_of_unit_incl_pad'] = (self._interval + 2 * self._pad)** 2 * self.N
        self.stat['no_neighbors'] = format(len(self.NEIGHBOR))
        self.stat['no_neighbor_point_min'] = np.min(result)
        self.stat['no_neighbor_point_max'] = np.max(result)
        self.stat['no_neighbor_point_mean'] = np.mean(result)
        self.stat['no_neighbor_point_std'] = statistics.stdev(result)        
        self.stat['no_actual_point_in_unit_min'] = np.min(unit_result)
        self.stat['no_actual_point_in_unit_max'] = np.max(unit_result)
        self.stat['no_actual_point_in_unit_mean'] = np.mean(unit_result)
        self.stat['no_actual_point_in_unit_std'] = statistics.stdev(unit_result)
        self.stat['no_actual_point_in_unit_incl_pad_min'] = np.min(pad_result)
        self.stat['no_actual_point_in_unit_incl_pad_max'] = np.max(pad_result)
        self.stat['no_actual_point_in_unit_incl_pad_mean'] = np.mean(pad_result)
        self.stat['no_actual_point_in_unit_incl_pad_std'] = statistics.stdev(pad_result)                
        # print(unit_result)
      

        
        matrix_size = 0
        for key in self._DIST.keys() : 
            matrix_size += self._DIST[key].shape[0] * self._DIST[key].shape[1]
        
        self.stat['matrix_original_size'] = len(self.data)**2
        self.stat['matrix_reduced_size'] = matrix_size
        self.stat['factor'] = round(self.N**2 / matrix_size,2)
        self.flag['compute_stats'] = True

    def print_stats(self, options: list = [True,True,True,True]) : 
        print("="*100)
        if options[0] : 
            print("Basic information")    
            print("\tSize of Points(stations){}".format(self.N))
            print("\tSize of Distance matrix dictionary: {0}".format(len(self._DIST)))
            print("\t# of points in unit: {}".format(self._no_of_points_in_unit))
            print("\t# of neighbors for computations {}".format(self._no_of_neighbors))
            print("\tmin and max in x and y", self._x_min, self._x_max, self._y_min, self._y_max)
            print("\tlength of x and y", self._lx, self._ly)
            print("\tinterval: {}\tPadding: {}".format(self._interval, self._pad))
            print("\t# of blocks in x and y", self._no_of_blocks_in_x, self._no_of_blocks_in_y)
            print("\tarea_of_unit:",self.stat['area_of_unit'])
            print("\texpected_no_of_points_by_area_of_unit: ", self.stat['expected_no_of_points_by_area_of_unit'])
            print("\tarea_of_unit_incl_padding: " ,self.stat['area_of_unit_incl_padding'] )
            print("\texpected_no_of_points_by_area_of_unit_incl_pad: ",self.stat['expected_no_of_points_by_area_of_unit_incl_pad'])
            print("\tno_actual_point_in_unit_min'",self.stat['no_actual_point_in_unit_min'] )
            print("\tno_actual_point_in_unit_max'",self.stat['no_actual_point_in_unit_max'] )
            print("\tno_actual_point_in_unit_mean",self.stat['no_actual_point_in_unit_mean'])
            print("\tno_actual_point_in_unit_std'",self.stat['no_actual_point_in_unit_std'] )
            print("\tno_actual_point_in_unit_incl_pad_min: ",self.stat['no_actual_point_in_unit_incl_pad_min'])
            print("\tno_actual_point_in_unit_incl_pad_max: ",self.stat['no_actual_point_in_unit_incl_pad_max'])
            print("\tno_actual_point_in_unit_incl_pad_mean: ",self.stat['no_actual_point_in_unit_incl_pad_mean'])
            print("\tno_actual_point_in_unit_incl_pad_std: ",self.stat['no_actual_point_in_unit_incl_pad_std'])
            # print("\tlist of x: ", self._lst_x)
            # print("\tlist of y: ",self._lst_y)

        if options[1] : 
            # minimum padding width = 1/2 * sqrt(k/N * lx * ly)        
            # interval is setout     
            
            print("-"*100)
            print("diagnosized over the NEIGHBOR dictionary")    
            print("\tSize of NEIGHBOR dictionary: {0}".format(len(self.NEIGHBOR)))
            print("\tnumber of points - min : {0}, max : {1}, mean: {2}, std: {3}".format(
                self.stat['no_neighbor_point_min'],
                self.stat['no_neighbor_point_max'],
                self.stat['no_neighbor_point_mean'],
                self.stat['no_neighbor_point_std']
            ))
        if options[2] : 
            print("-"*100)
            print("Matrix Size Efficiency")    
            print("\tmatrix original size: {0}, \n\tmatrix reduced size : {1}, \n\tfactor of save {2}".format(
                self.stat['matrix_original_size'],
                self.stat['matrix_reduced_size'],
                self.stat['factor'],
            ))
        if options[3] : 
            print("-"*100)
            print("Runtime")    
            print("\ttotal run time: {0} \n\t\tDist_matrix: {1} \n\t\tFind_neighbor: {2}".format(
                round(self.stat['elapsed_time_create_dist_matrix'] +
                self.stat['elapsed_time_find_neighbor'],3),
                round(self.stat['elapsed_time_create_dist_matrix'],3),
                round(self.stat['elapsed_time_find_neighbor'],3)
            ))
        print("="*100)    

    def pick_and_draw_random_block(self) :                      
        while True: 
            random_block_idx = random.choice(list(self._IDX_w_padding.keys()))
            points = self._IDX_w_padding[random_block_idx]
            mask = self._IDX_masking[random_block_idx]

            if len(points[mask]) > 0 : 
                break
        padding_points = self.data[points]
        center_points = self.data[points[mask]]
        target_point = random.choice(points[mask])
        
        f, axs = plt.subplots(2,2,figsize=(15,30))

        
        ax1 = plt.subplot(1,2,1)
        ax2 = plt.subplot(1,2,2)

        self.draw(ax1, self.data, self.NEIGHBOR, random_block_idx, target_point, center_points, padding_points, False)
        self.draw(ax2, self.data, self.NEIGHBOR, random_block_idx, target_point, center_points, padding_points, True)
        
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
            

def test(N, n,k) : 
    print("\n"*2)
    print("="*100)
    print(N, n, k)
    data = np.random.rand(N,2)
    dm = DistanceMatrix(data, n,k)         
    dm.create_dist_matrix()
    dm.find_neighbor(k=10)    
    
    dm.compute_stats()
    dm.print_stats()
    
    # dm.pick_and_draw_random_block()
    print("="*100)
    result =   [
        dm.N,
        n,
        k,
        dm.stat['factor'],
        round(dm.stat['elapsed_time_create_dist_matrix'] + dm.stat['elapsed_time_find_neighbor'],3),
        round(dm.stat['elapsed_time_create_dist_matrix'],3),
        round(dm.stat['elapsed_time_find_neighbor'],3),
        len(dm._DIST),
        dm._no_of_points_in_unit,
        dm._no_of_neighbors,
        dm._x_min, 
        dm._x_max, 
        dm._y_min, 
        dm._y_max,
        dm._lx, 
        dm._ly,
        dm._interval, 
        dm._pad,
        dm._no_of_blocks_in_x, 
        dm._no_of_blocks_in_y,
        dm.stat['area_of_unit'],
        dm.stat['expected_no_of_points_by_area_of_unit'],
        dm.stat['area_of_unit_incl_padding'],
        dm.stat['expected_no_of_points_by_area_of_unit_incl_pad'],
        dm.stat['no_actual_point_in_unit_min'] ,
        dm.stat['no_actual_point_in_unit_max'] ,
        dm.stat['no_actual_point_in_unit_mean'], #-13
        dm.stat['no_actual_point_in_unit_std'], 
        dm.stat['no_actual_point_in_unit_incl_pad_min'],
        dm.stat['no_actual_point_in_unit_incl_pad_max'], 
        dm.stat['no_actual_point_in_unit_incl_pad_mean'], #-9
        dm.stat['no_actual_point_in_unit_incl_pad_std'], 
        dm.stat['no_neighbor_point_min'], #-7
        dm.stat['no_neighbor_point_max'], #-6 
        dm.stat['no_neighbor_point_mean'], # -5
        dm.stat['no_neighbor_point_std'], #-4
        dm.stat['matrix_original_size'], #-3
        dm.stat['matrix_reduced_size'], #-2
        dm.stat['factor'] #-1
    ]
    filename = 'test_result.txt'
    f = open(filename, 'a')    
    
    f.write("\n")
    f.write(str(result)[1:-1])
    f.close()


if __name__ == '__main__' : 
    N = int(sys.argv[1])      
    

    
    test(int(N), n=100, k=10)
    

    # def handler(signum, frame):
    #     print("timeout")
    #     raise Exception("end of time")

    # for n in N : 
    #     for s in S : 
    #         for p in P : 
    #             if n <= s : 
    #                 continue      
    #             signal.signal(signal.SIGALRM, handler)
    #             signal.alarm(300)
    #             try : 
    #                 test(n,s,p)
    #             except Exception as e:
    #                 print(e)


    # print(DIST)
    


    # distmat.pick_and_draw_random_block()
    


    
    #     print(metadata.data.shape)
    #     DIST = distance_matrix(metadata.data, metadata.data)
    #     print(DIST.shape)
    #     end = time.time() - start
    #     print("\n", i, end)
    #     result.append((i, end))
    
    # dist_matrix_creator


    