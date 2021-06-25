
from abc import abstractmethod, ABC
from util import levenshteinDistance
import copy
import numpy as np
from sklearn.linear_model import LinearRegression


from data_generator import Data
class Predictor(ABC) : 
    pass
class SpatialComparision(Predictor) :
    pass

class EnsembleRegression(Predictor):
    pass


class SmoothingAndPredict(Predictor):    
    def predict(self, data: Data) : 
        ts_data = data.ts_data
        neighbor = data.neighbor
        sum_edit_distance = 0
        for i in range(len(data.ts_data)) : 
            try : 
                answer = sorted(data.dic_timesteps[i])
            except :
                answer = []
            y = copy.deepcopy(ts_data[i])
            X = copy.deepcopy(ts_data[neighbor[i]])
            corrected_y = self.correct(X,y)                                    
            suspected_timesteps = sorted(np.where(abs(ts_data[i] - corrected_y)>3)[0])            
            edit_distance = levenshteinDistance(answer,suspected_timesteps)
            sum_edit_distance+= edit_distance
            # print(i, edit_distance)
        return sum_edit_distance

    def correct(self, X, y) : 
        while True :  
            reg = LinearRegression().fit(X.T, y)
    #         print(reg.score(X.T, y))
            pred = reg.predict(X.T)
            error = pred - y 
            error_mean = np.mean(error)
            error_std = np.std(error)
            idx_boolean = (error >= error_mean +  max(2 * error_std,3)) | (error <= error_mean -  max(2 * error_std,3))
            idx = np.where(idx_boolean == True )[0]
            y[idx] = pred[idx]  
            
            if len(idx) == 0 :            
                break
        return y

    


class RegressionWithNeighbor(Predictor):
    def predict(ts_data, i) :     
        y = ts_data[i]
        X = ts_data[neighbor[i]]
        reg = LinearRegression().fit(X.T,y)
        score = reg.score(X.T,y)
        print(score)
        pred = reg.predict(X.T)
        gap2 = abs(pred - y)
        plt.figure(figsize=(20,5))
        plt.plot(X.T, color='grey')
        plt.plot(y)
        plt.plot(pred, color='black')
        # plt.plot(gap, color='blue')
        return pred
class RegressionBestSelectionByRandom(Predictor) : 
    def predict(ts_data, i, global_search=False, display=False) :     
        max_score = -np.inf
        max_iteration = 50
        n_iter = 0 
        thr = 0.9
        while True :
            if global_search : 
                candidate = np.arange(ts_data.shape[0])
                candidate = np.delete(candidate, i)                      
            else : 
                candidate = data._neighbor[i]      
            idx_stations = np.random.choice(candidate, size=3, replace=False)
            idx_timesteps = np.random.choice(np.arange(ts_data.shape[1]), size= round(ts_data.shape[1]), replace=True)
            idx_timesteps.sort()

            y = ts_data[i]
            X = ts_data[idx_stations]
            reg = LinearRegression().fit(X.T[idx_timesteps],y[idx_timesteps])
            score = reg.score(X.T[idx_timesteps],y[idx_timesteps])


            if score > max_score :
                max_score = score
                max_idx_stations = idx_stations
                max_idx_timesteps = idx_timesteps

            if n_iter >= max_iteration :                 
                X = ts_data[max_idx_stations]
                reg = LinearRegression().fit(X.T[max_idx_timesteps],y[max_idx_timesteps])
                pred = reg.predict(X.T)
                gap = abs(pred - y)
                if display : 
                    print("for {} iteration, cannot find more than {}. the best upto now is {}".format(max_iteration, thr, max_score))
                break
            if (score > thr) : 
                pred = reg.predict(X.T)
                gap = pred - y
                if display : 
                    print(score)
                break
            n_iter += 1
        if display : 
            plt.figure(figsize=(20,3))
            plt.plot(X.T, color='grey')
            plt.plot(y)
            plt.plot(pred, color='black')
            # plt.plot(gap, color='blue')
        return pred