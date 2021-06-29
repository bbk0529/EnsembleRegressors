

import copy
import numpy as np
import pickle
from abc import abstractmethod, ABC
import random
import pandas as pd
from metadata import DWDLocation

class Data(ABC):
    pass

class RandomDataGenerator(Data) : 
    pass


class Data(ABC):    
    def select_data(self, n_stations: int, n_timesteps: int) :        
        self.ts_data = self.preprocess_data(self.ts_data[:n_stations, :n_timesteps]   )       
        
    

class RandomData(Data) : 
    def __init__(self, n_stations: int, n_timesteps: int, k: int=5, p_noise: float=None) :
        self._metadata = self._create_metadata(n_stations)
        self._ts_rawdata = self._create_temperature_data(n_stations, n_timesteps)         
        self._k = k        
        super().__init__(p_noise)

        

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
    def __init__(self, filename: str='data_dvd_reduced.p', ) : 
        df =  pickle.load(open(filename,'rb'))       
        self.ts_data = self.preprocess_data(df.values)        

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

if __name__ == '__main__' :
    data = DWDTemperatureLoader()
    print(data)