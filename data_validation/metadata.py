import numpy as np
import pickle
from scipy.spatial import distance_matrix
import random

class Metadata():
    def create_neighor_list(self, metadata: np.ndarray): 
        #neighbor list
        self.dist_matrix = distance_matrix(metadata, metadata)
        self.neighbor = self.dist_matrix.argsort()
        return self.dist_matrix.argsort()

class RandomMetadata(Metadata) : 
    def __init__(self, n_stations: int,height: int = 100, latitude: int = 100, longitude:int = 100) : 
        #random metadata creation 
        metadata = np.zeros((n_stations,3), dtype='int')
        for i in range(n_stations) : 
            z = random.choice(range(height))
            lat = random.choice(range(latitude))
            lon = random.choice(range(longitude))
            metadata[i] = [z, lat, lon]
        self.locations = metadata


class DWDLocation(Metadata) :   
    def __init__(self, filename: str='metadata.p',  k=5): 
        df = pickle.load(open(filename, 'rb'))            
        self.locations = df.values     
        self.create_neighor_list(self.locations[:,1:])
