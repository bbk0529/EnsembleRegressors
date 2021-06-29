import numpy as np

class Classifier() :
    pass

class ThresholdClassifier(Classifier) : 
    def __init__(self, eps) : 
        self.eps = eps
    
    def classify(self, ts_actual: np.array, ts_predict: np.array) : 
        suspected_timesteps = sorted(np.where(abs(ts_actual - ts_predict)>self.eps)[0])                    
        return suspected_timesteps



