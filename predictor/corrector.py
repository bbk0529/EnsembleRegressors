from sklearn.linear_model import LinearRegression

from abc import abstractmethod, ABC
import numpy as np

class Corrector(ABC) : 
    pass

class regressionCorrector(Corrector) :
    def __init__(self, X: np.ndarray, y: np.array) : 
        self.X = X
        self.y = y
    
    def correct(self, X, y, min_value=3) : 
        while True :  
            reg = LinearRegression().fit(X.T, y)    
            pred = reg.predict(X.T)
            error = pred - y 
            error_mean = np.mean(error)
            error_std = np.std(error)
            idx_boolean = (error >= error_mean +  max(2 * error_std, min_value)) | (error <= error_mean -  max(2 * error_std, min_value))
            idx = np.where(idx_boolean == True )[0]
            y[idx] = pred[idx]

            if len(idx) == 0 :            
                break
        return y

    