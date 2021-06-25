import copy
import numpy as np
import pickle
from abc import abstractmethod, ABC
from scipy.spatial import distance_matrix
import random
import pandas as pd

class Data(ABC):
    pass

class RandomDataGenerator(Data) : 
    pass


class Data(ABC):
    def __init__(self, p_noise_stations, p_noise_timesteps):
        self._create_neighor_list(self._metadata[:, 1:], self._k)                        
        ts_rawdata = copy.deepcopy(self._ts_rawdata)        
        self.ts_data, self.lst_station_timestep = self.add_noise3(ts_rawdata, p_noise_stations=p_noise_stations, p_noise_timesteps = p_noise_timesteps)
        
    def _create_neighor_list(self, metadata: np.ndarray, k: int): 
        #neighbor list
        self._dist_matrix = distance_matrix(metadata, metadata)
        self.neighbor = self._dist_matrix.argsort()[:, 1:k+1]        

    def add_noise2(self, ts_data, p_noise: float) :
        matrix_noises = np.random.rand(ts_data.shape[0], ts_data.shape[1])
        pick = np.random.choice(np.arange(0, 1, 0.1))
        idx_null = (matrix_noises > pick + p_noise) | (matrix_noises <= pick)
        idx_positve = matrix_noises> pick + p_noise/2
        idx_negative = matrix_noises<= pick + p_noise/2
        matrix_noises[idx_positve] = 1
        matrix_noises[idx_negative] = -1        
        matrix_noises[idx_null] = 0
        
        matrix_noises = matrix_noises * np.random.rand(ts_data.shape[0], ts_data.shape[1]) * 5         
        idx = np.where(abs(matrix_noises)>0)
        lst_noises ={}
        return ts_data + matrix_noises, [str(x[0]) + '_' + str(x[1]) for x in np.array(idx).T]

    def add_noise3(self, ts_data, p_noise_stations: float, p_noise_timesteps: float) :
        n_stations = ts_data.shape[0]
        n_timesteps = ts_data.shape[1]
        self.dic_timesteps ={}
        matrix_noises = np.zeros(ts_data.shape)
        picked_stations = np.random.choice(range(n_stations), size= round(p_noise_stations * n_stations), replace=False)
        picked_stations.sort()
        self._picked_stations = picked_stations
        for s in picked_stations : 
            pick_timesteps = np.random.choice(range(n_timesteps), size= round(p_noise_timesteps * n_timesteps), replace=False)
            # matrix_noises[s, pick_timesteps ] += 5 * np.random.choice([-1,1])
            matrix_noises[s, pick_timesteps ] = np.random.rand(len(pick_timesteps)) + random.choice(range(3,5)) * np.random.choice([-1,1])
            self.dic_timesteps[s] = pick_timesteps
            # matrix_noises[s, pick_timesteps ] = np.random.choice(np.arange(3,5,1)) * np.random.choice([-1,1]) - np.random.randn(len(pick_timesteps)) * 10 
        idx = np.where(abs(matrix_noises)>0)
        lst_noises ={}
        self.matrix_noises = matrix_noises
        return ts_data + matrix_noises, [str(x[0]) + '_' + str(x[1]) for x in np.array(idx).T]



    def add_noise(self, ts_data, p_noise: float, min_noise=1, max_noise=10) :
        n_row = ts_data.shape[0]
        n_col = ts_data.shape[1]
        lst_station_timestep = []
        lst_noises = {}

        while(ts_data.size * p_noise > len(lst_noises)) : 
            #selecting index and create noise
            failed_station = random.choice(range(n_row))
            failed_timestep = random.choice(range(n_col))            
            noise = max(min(random.random()*20, max_noise), min_noise) * random.choice([-1,1])            
            
            #adding noise 
            ts_data[failed_station, failed_timestep] = ts_data[failed_station, failed_timestep] + noise                        
            
            #creating data for further works
            lst_noises[failed_station, failed_timestep] = round(noise,3)
            lst_station_timestep.append(str(failed_station) + "_" + str(failed_timestep))
            lst_station_timestep = list(set(lst_station_timestep))
            lst_station_timestep.sort()
        self._flag_noise = True
        return ts_data, lst_station_timestep, lst_noises


    

class RandomData(Data) : 
    def __init__(self, n_stations: int, n_timesteps: int, k: int=5, p_noise: float=None) :
        self._metadata = self._create_metadata(n_stations)
        self._ts_rawdata = self._create_temperature_data(n_stations, n_timesteps)         
        self._k = k        
        super().__init__(p_noise)

        

    def _create_metadata(self, n_stations: int,height: int = 100, latitude: int = 100, longitude:int = 100) : 
        #random metadata creation 
        metadata = np.zeros((n_stations,3), dtype='int')
        for i in range(n_stations) : 
            z = random.choice(range(height))
            lat = random.choice(range(latitude))
            lon = random.choice(range(longitude))
            metadata[i] = [z, lat, lon]
        return metadata

    def _create_temperature_data(self, n_stations, n_timesteps) :
        #random data creation 
        ts_data = np.zeros((n_stations, n_timesteps))
        lst_failed_station = {}
        
        for time_step in range(n_timesteps) :                    
            s = random.choice(range(-4,40,1))
            val = s + np.random.rand(n_stations) * 5               
            ts_data[:,time_step] = val
        # print(len(lst_failed_station))
        return ts_data        


class DWDTemperatureLoader(Data) : 
    def __init__(self, n_stations: int, n_timesteps: int, k: int=5, p_noise_stations: float=0.1, p_noise_timesteps: float=0.1) : 
        df = pickle.load(open('../data_dvd_reduced.p','rb'))
        df_metadata = pickle.load(open('../metadata.p', 'rb'))            
        self._metadata = df_metadata.values[:n_stations]        
        self._ts_rawdata = self.preprocess_data(df.values[:n_stations, :n_timesteps]   )
        self._k = k          
        super().__init__(p_noise_stations, p_noise_timesteps)

    def preprocess_data(self, ts_data: np.ndarray) : 
        df = pd.DataFrame(ts_data)
        df[df<=-40] = np.nan
        df[df>50] = np.nan
        df = df.fillna(method='bfill')
        df = df.fillna(method='ffill')
        return df.values
   



class IntermediateDataStore(Data):
    pass
class CorrectedDataStore(Data):
    pass