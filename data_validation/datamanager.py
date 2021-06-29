from ts_data import *
from metadata import *
class DataManager() :
    def __init__(self, data_instance:Data, metadata_instance:Metadata):
        self.data_instance = data_instance
        self.metadata_instance = metadata_instance
        self.ts_data = self.data_instance.ts_data
        self.neighbor = self.meta_instance.create_neighor_list
    
    def select_data(self, n_stations: int, n_timesteps: int) :        
        self.ts_data = self.ts_data.preprocess_data(self.ts_data[:n_stations, :n_timesteps])
        self.metadata = 