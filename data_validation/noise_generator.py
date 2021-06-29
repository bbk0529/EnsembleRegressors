import numpy as np
from abc import abstractmethod, ABC


  

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



class NoiseGenerator(ABC) :     
    def __init__(self, uniform = True, min_noise: float=-np.inf, max_noise: float=np.inf): 
        self.uniform = uniform        
        self.min_noise = min_noise
        self.max_noise = max_noise
        

    @abstractmethod
    def generate_noises():
        pass

class ClippedNoises(NoiseGenerator) : 
    def __init__(self, distribution, min, max):
        super().__init__(distribution, min, max)

    def generate_noises(self, n_stations, n_timesteps,  p_noise_stations: float, p_noise_timesteps: float):        
                
        dic_timesteps ={}
        matrix_noises = np.zeros((n_stations, n_timesteps))
        picked_stations = np.random.choice(range(n_stations), size= round(p_noise_stations * n_stations), replace=False)
        picked_stations.sort()
        self._picked_stations = picked_stations
        
        for s in picked_stations : 
            pick_timesteps = np.random.choice(range(n_timesteps), size= round(p_noise_timesteps * n_timesteps), replace=False)            
            noises = np.append(
                np.random.rand(round(len(pick_timesteps)/2)) * self.max_noise, 
                - np.random.rand(len(pick_timesteps) - round(len(pick_timesteps)/2)) * self.max_noise
            )        
            # clipping 
            noises[(0 <= noises) & (noises <self.min_noise) ] = np.random.choice(range(self.min_noise,self.max_noise))
            noises[(0 >= noises) & (noises > -self.min_noise) ] = np.random.choice(range(-self.max_noise,-self.min_noise))
            
            # noise injection
            matrix_noises[s, pick_timesteps] = noises            
            dic_timesteps[s] = pick_timesteps        
        
        
        
        self.dic_timesteps = dic_timesteps
        self.matrix_noises = matrix_noises
        idx_noises = np.array(np.where(abs(matrix_noises)>0)).T
        self.idx_noises = tuple(map(tuple, idx_noises))

        return matrix_noises 
        # return matrix_noises, [str(x[0]) + '_' + str(x[1]) for x in np.array(idx_noises).T]

        

class Noises :
    pass


if __name__ == "__main__" : 

    ts_data = np.zeros((100,100))
    noise = ClippedNoises(True, min=3, max=10)
    matrix_noises = noise.generate_noises(ts_data.shape[0], ts_data.shape[1], p_noise_stations = 0.2, p_noise_timesteps = 0.05)
    print(matrix_noises)