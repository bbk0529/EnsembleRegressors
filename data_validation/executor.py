from data_generator import Data
from predictor import Predictor
from corrector import Corrector
from data_generator import DWDTemperatureLoader

from predictor import SmoothingAndPredict
class Executor() : 
    def __init__(self, data: Data, predictor: Predictor, corrector: Corrector = None):
        pass
        self.data = data
        self.predictor = predictor
        self.corrector = corrector

    def validate_single_station(self, station) : 
        self.predictor.predict(self.data, station)
    
    def validate_whole_stations(self) : 
        sum_edit_distances = self.predictor.predict(self.data)
        print(sum_edit_distances)

if __name__ == '__main__' : 
    n_stations = 450
    n_timesteps= 50 #maybe 288 is the maximum to propoerly manage them 
    k = 5
    p_noise_stations = 0.05
    p_noise_timesteps= 0.05
    print(n_stations * n_timesteps * p_noise_stations * p_noise_timesteps, n_stations * n_timesteps)

    data = DWDTemperatureLoader(n_stations, n_timesteps, k, p_noise_stations=p_noise_stations, p_noise_timesteps= p_noise_timesteps)
    predictor = SmoothingAndPredict()
    executor = Executor(data, predictor)
    
    executor.validate_whole_stations()
    

    