
from classifier import *
from noise_generator import *
from data_generator import *
from predictor import *
from abc import abstractmethod, ABC


def evaluate_predictor(function, ts_data, i, global_search=False, display = False) : 
    pred = function(ts_data, i, global_search=global_search, display=display)

    gap = pred - ts_data[i]
    gap_between_rawdata = pred - data._ts_rawdata[i]
    
    gap_mean = np.mean(gap)
    gap_std = max(np.std(gap),2)

    idx = (gap >= gap_mean +  2 * gap_std) | (gap <= gap_mean -  2 * gap_std)
    pred_by_neighbor_randomness = np.where(idx == True )[0]
    
    if display  : 
        print("SSE between rawdata", math.sqrt(sum(gap_between_rawdata**2)))
        print(pred_by_neighbor_randomness)
        print("LevenshteinDistance", levenshteinDistance(answer, pred_by_neighbor_randomness))
        plt.plot(idx, color='red')
        
        
        plt.figure(figsize=(20,3))
        plt.plot(pred, color='black')        
        plt.plot(data._ts_rawdata[i], color='green')
        
    return levenshteinDistance(answer, pred_by_neighbor_randomness)
class Evaluator(ABC) :
    pass 


class ConnfusionMatrix(Evaluator) : 
    pass
    """
    
    F1 SCORE



    """

class AllStationsEvaluator(Evaluator) : 
    def __init__(self, predictor: Predictor, classifier: Classifier, data: Data, metadata: Metadata, noise_generator: NoiseGenerator) : 
        self.predictor = predictor
        self.classifier = classifier
        self.data = data #rawdata        
        self.ts_data = data.ts_data #working data before adding noises by default
        self.noise_generator = noise_generator        
        self.neighbor = metadata.neighbor        
        self.toggle_noises = False
        
    def add_noises(self, p_noise_stations, p_noise_timesteps) : 
        n_stations = self.ts_data.shape[0]
        n_timesteps = self.ts_data.shape[1]
        matrix_noises = self.noise_generator.generate_noises(n_stations, n_timesteps,  p_noise_stations, p_noise_timesteps)
        self.test_data = self.ts_data + matrix_noises        
        self.toggle_noises = True

    def select_data(self, n_stations, n_timesteps ) :        
        n_total_stations = self.data.ts_data.shape[0]
        n_total_timesteps = self.data.ts_data.shape[1]
        self.idx_stations = np.random.choice(range(n_total_stations), size=n_stations)        
        idx_timesteps_start = np.random.choice(range(0, n_total_timesteps - n_timesteps))
        self.idx_timesteps = slice(idx_timesteps_start, idx_timesteps_start + n_timesteps)              
    

    
    def evaluate(self) :
        n_stations = 450
        n_timesteps = 100
        k = min(round(n_stations/5),10)
        self.select_data(n_stations,n_timesteps)                        
        ts_data = self.ts_data
        neighbor = self.neighbor

        

        for station in range(n_stations) : 
            ts_actual = ts_data[station]
            ts_predict = self.predictor.predict(ts_data, neighbor, station)            
            suspected_timesteps = self.classifier.classify(ts_actual, ts_predict)
            print(suspected_timesteps)

        


if __name__ == '__main__' : 
    predictor = SmoothingAndPredict()
    classifier = ThresholdClassifier(eps = 3)
    data = DWDTemperatureLoader()
    metadata = DWDLocation()
    noise_generator = ClippedNoises(distribution=True, min=3, max=10)
    evaluator = AllStationsEvaluator(predictor, classifier, data, metadata, noise_generator )
    evaluator.evaluate()