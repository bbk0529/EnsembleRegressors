
from __future__ import annotations
import time
import math
import numpy as np
import pandas as pd

import random
from sklearn.linear_model import LinearRegression
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import pickle
import copy

from abc import abstractmethod, ABC

# def spatialcomparision(ts_data: pd.DataFrame, neighbor_data: pd.DataFrame, station: str) : 

class Classifier(ABC): 
    @abstractmethod
    def validate() :
        pass






class SpatialComparision(Classifier) :
    def validate(self, ts_data: np.ndarray, neighbor: np.ndarray, station: int, n_regressors = None,  n_variables = None, eps = None  ):
        v_col = 'value'
        neighbor_data = pd.DataFrame(ts_data[neighbor[station]].T)
        ts_data = ts_data[station]
        ts_data = pd.DataFrame(ts_data, columns = [v_col])
        

        const_rel_dw = 0.1
        const_rel_up = 0.1
        size = len(ts_data)
        
        const_abs = np.nan
        value_min = -np.inf
        value_max = np.inf
        
        aux_matrix = np.zeros((size, 3))
        aux_matrix[:, 0] = neighbor_data.mean(axis=1).abs() * const_rel_dw
        aux_matrix[:, 1] = neighbor_data.mean(axis=1).abs() * const_rel_up
        aux_matrix[:, 2] = [const_abs] * size
        aux_matrix[:, 0] = np.nanmax(aux_matrix[:, [0, 2]], axis=1)
        aux_matrix[:, 1] = np.nanmax(aux_matrix[:, [1, 2]], axis=1)
        aux_matrix[:, 2] = neighbor_data.std(axis=1) * 2
        aux_matrix[:, 0] = np.nanmax(aux_matrix[:, [0, 2]], axis=1)
        aux_matrix[:, 1] = np.nanmax(aux_matrix[:, [1, 2]], axis=1)

        ts_data = ts_data.assign(
            min=neighbor_data.min(axis=1) - aux_matrix[:, 0],
            max=neighbor_data.max(axis=1) + aux_matrix[:, 1],
        )
        event_time_index = ts_data[
            ((value_min <= ts_data[v_col]) & (ts_data[v_col] <= value_max))
            & ((ts_data[v_col] < ts_data["min"]) | (ts_data[v_col] > ts_data["max"]))
        ].index

        return [str(station) + "_" + str(x) for x in event_time_index]


class EnsembleRegression(Classifier) :
    def __init__(self, n_regressors: int, n_variables: int, eps: float, decision_boundary: float, global_search = False) :
        self._n_regressors = n_regressors
        self._n_variables = n_variables
        self._eps = eps
        self._decision_boundary = decision_boundary
        self._global_search = global_search

    def validate(self, ts_data, neighbor, station) :
        result = np.array([], dtype='int')
        for i in range(self._n_regressors) : 
            if self._global_search : 
                idx = np.random.choice(range(len(ts_data)), size=self._n_variables, replace=False)
            else : 
                idx = np.random.choice(neighbor[station], size=self._n_variables, replace=False)
            
            y = ts_data[station] 
            X = ts_data[idx]
            reg = LinearRegression().fit(X.T, y)
            # print(reg.score(X.T, y))
            original = ts_data[station]
            predict = reg.predict(X.T)
            # predict = reg.intercept_ + np.dot(X.T, reg.coef_)
            ix = np.where(abs(predict - original) > self._eps)    
            result = np.append(result, ix)    

        unique, counts = np.unique(result, return_counts=True)
        RESULT = dict(zip(unique, counts))
        # print(RESULT)
        # return [str(station) + '_' + str(k) for k, v in sorted(RESULT.items(), reverse=True, key=lambda item: item[1]) if v>(n_regressors/3*2)]
        return [str(station) + '_' + str(k) for k, v in RESULT.items() if v>(self._n_regressors * self._decision_boundary)]

    def validate_and_predict(self, ts_data, neighbor, station):    
        width = ts_data.shape[1]
        PRED = np.zeros((self._n_regressors, width))
        SCORE = np.zeros(self._n_regressors)
        for j in range(self._n_regressors) :             
            idx_variables = np.random.choice(range(self._n_variables), replace=False, size=self._n_variables)
            y = ts_data[station]
            X = ts_data[neighbor[station]][idx_variables]
            reg = LinearRegression().fit(X.T,y)
            score = reg.score(X.T, y)
            pred = reg.predict(ts_data[neighbor[station]][idx_variables].T)            
            PRED[j] = pred
            SCORE[j] = score
        SCORE = SCORE / np.sum(SCORE)
        predict = np.dot(PRED.T, SCORE)
        original = ts_data[station]
        ix = np.where(abs(predict - original) > self._eps)
        if len(ix[0]) <= 0 : 
            return None
        else : 
            return [ix, pred[ix]]
    

class EnsembleRegression2(Classifier) :
    def __init__(self, n_regressors: int, n_variables: int, eps: float, decision_boundary: float) :
        self._n_regressors = n_regressors
        self._n_variables = n_variables
        self._eps = eps
        self._decision_boundary = decision_boundary

    def validate(self, ts_data, neighbor, station) :
        result = np.array([], dtype='int')
        for i in range(self._n_regressors) : 
            width = ts_data.shape[1]
            idx_samples = np.unique(np.random.choice(range(width), replace=True, size=width))
            idx =  np.random.choice(neighbor[station], size=self._n_variables, replace=False)
            y = ts_data[station][idx_samples]
            X = ts_data[idx][:, idx_samples]
            reg = LinearRegression().fit(X.T,y)            
            original = ts_data[station]
            predict = reg.predict(ts_data[idx].T)                
            ix = np.where(abs(predict - original) > self._eps)    
            result = np.append(result, ix)    

        unique, counts = np.unique(result, return_counts=True)
        RESULT = dict(zip(unique, counts))
        # print(RESULT)
        # return [str(station) + '_' + str(k) for k, v in sorted(RESULT.items(), reverse=True, key=lambda item: item[1]) if v>(n_regressors/3*2)]
        return [str(station) + '_' + str(k) for k, v in RESULT.items() if v>(self._n_regressors * self._decision_boundary)]

class EnsembleRegressionBootstrap(Classifier) :
    def __init__(self, n_regressors: int, n_variables: int, eps: float) :
        self._n_regressors = n_regressors
        self._n_variables = n_variables
        self._eps = eps        

    def validate(self, ts_data, neighbor, station) :
        width = ts_data.shape[1]
        PRED = np.zeros((self._n_regressors, width))
        SCORE = np.zeros(self._n_regressors)
        for j in range(self._n_regressors) : 
            idx_samples = np.unique(np.random.choice(range(len(ts_data[station])), replace=True, size=40))
            idx_variables = np.random.choice(range(self._n_variables), replace=False, size=self._n_variables)
            y = ts_data[station][idx_samples]
            X = ts_data[neighbor[station]][idx_variables][:, idx_samples]
            reg = LinearRegression().fit(X.T,y)
            score = reg.score(X.T, y)
            pred = reg.predict(ts_data[neighbor[station]][idx_variables].T)            
            PRED[j] = pred
            SCORE[j] = score
        SCORE = SCORE / np.sum(SCORE)
        predict = np.dot(PRED.T, SCORE)
        original = ts_data[station]
        ix = np.where(abs(predict - original) > self._eps)    
        return [str(station) + '_' + str(idx) for idx in ix[0]]

class SmoothingAndPredict(Classifier):    
    def correct(self, X, y, graph=False) : 
        ymax = max(abs(y))
        while True :  
            reg = LinearRegression().fit(X.T, y)    
            if graph : 
                plt.figure(figsize=(20,3))
                plt.ylim([-ymax, ymax])
                plt.plot(y)
            pred = reg.predict(X.T)
            error = pred - y 
            error_mean = np.mean(error)
            error_std = np.std(error)
            # idx_boolean = (error >= error_mean +  max(1.5 * error_std) | (error <= error_mean -  1.5 * error_std)
            idx_boolean = (error >= error_mean +  max(2 * error_std,2)) | (error <= error_mean -  max(2 * error_std,2))
            idx = np.where(idx_boolean == True )[0]
            y[idx] = pred[idx] 
            if graph : 
                
                plt.plot(y)
                plt.legend(['data_w_noises','corrected'])
            
            if len(idx) == 0 :            
                break
        return y
    
    def validate(self, ts_data, neighbor, station) :               
        y = copy.deepcopy(ts_data[station])
        X = copy.deepcopy(ts_data[neighbor[station]])
        corrected_y = self.correct(X,y)                                    
        suspected_timesteps = sorted(np.where(abs(ts_data[station] - corrected_y)>3)[0])                   
        return [str(station) + '_' + str(idx) for idx in suspected_timesteps]



class RansacRegressor(Classifier) : 
    def validate (self, ts_data, neighbor, station): 
        from sklearn.linear_model import RANSACRegressor
        model = RANSACRegressor()
        X = ts_data[neighbor[station]]
        y = ts_data[station]
        model.fit(X.T, y)
        suspected_timesteps = sorted(np.where(abs(model.predict(X.T) - y) > 3)[0])
        return [str(station) + '_' + str(idx) for idx in suspected_timesteps]


class HuberRegressor(Classifier) : 
    def validate (self, ts_data, neighbor, station): 
        from sklearn.linear_model import HuberRegressor
        model = HuberRegressor(max_iter=10)
        X = ts_data[neighbor[station]]
        y = ts_data[station]
        model.fit(X.T, y)
        suspected_timesteps = sorted(np.where(abs(model.predict(X.T) - y) > 3)[0])
        return [str(station) + '_' + str(idx) for idx in suspected_timesteps]

class TheilSenRegressor(Classifier) : 
    def validate (self, ts_data, neighbor, station): 
            from sklearn.linear_model import TheilSenRegressor
            model = TheilSenRegressor()
            X = ts_data[neighbor[station]]
            y = ts_data[station]
            model.fit(X.T, y)
            suspected_timesteps = sorted(np.where(abs(model.predict(X.T) - y) > 3)[0])
            return [str(station) + '_' + str(idx) for idx in suspected_timesteps]



class EnsembleRegressionCorrector(Classifier) :
    def __init__(self, n_regressors: int, n_variables: int, eps: float) :
        self._n_regressors = n_regressors
        self._n_variables = n_variables
        self._eps = eps        

    def validate(self, ts_data, neighbor, station) :
        width = ts_data.shape[1]
        PRED = np.zeros((self._n_regressors, width))
        SCORE = np.zeros(self._n_regressors)
        for j in range(self._n_regressors) : 
            idx_samples = np.unique(np.random.choice(range(len(ts_data[station])), replace=True, size=40))
            idx_variables = np.random.choice(range(self._n_variables), replace=False, size=self._n_variables)
            y = ts_data[station][idx_samples]
            X = ts_data[neighbor[station]][idx_variables][:, idx_samples]
            reg = LinearRegression().fit(X.T,y)
            score = reg.score(X.T, y)
            pred = reg.predict(ts_data[neighbor[station]][idx_variables].T)            
            PRED[j] = pred
            SCORE[j] = score
        SCORE = SCORE / np.sum(SCORE)
        predict = np.dot(PRED.T, SCORE)
        original = ts_data[station]
        ix = np.where(abs(predict - original) > self._eps)    
        ts_data[station, ix] = predict[ix]
        return [str(station) + '_' + str(idx) for idx in ix[0]]

class Data(ABC):
    def __init__(self, p_noise_stations, p_noise_timesteps, min_noises, max_noises):
        self._create_neighor_list(self._metadata[:, 1:], self._k)                        
        ts_rawdata = copy.deepcopy(self._ts_rawdata)        
        self.ts_data, self.lst_station_timestep = self.add_noise3(
            ts_rawdata, p_noise_stations=p_noise_stations, p_noise_timesteps = p_noise_timesteps, min_noises=min_noises, max_noises=max_noises
        )
        
        
        

    def _create_neighor_list(self, metadata: np.ndarray, k: int): 
        #neighbor list
        self._dist_matrix = distance_matrix(metadata, metadata)
        self._neighbor = self._dist_matrix.argsort()[:, 1:k+1]        


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

    def add_noise3(self, ts_data, p_noise_stations: float, p_noise_timesteps: float, min_noises, max_noises) :
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
            # matrix_noises[s, pick_timesteps ] = np.random.rand(len(pick_timesteps)) + random.choice(range(5,10)) * np.random.choice([-1,1])
            noises = np.append(
                np.random.rand(round(len(pick_timesteps)/2)) * max_noises, 
                - np.random.rand(len(pick_timesteps) - round(len(pick_timesteps)/2)) * max_noises
            )
            # noises = np.random.randn(len(pick_timesteps)) * 5
            noises[(0 <= noises) & (noises < min_noises) ] = np.random.choice(range(min_noises,max_noises))
            noises[(0 >= noises) & (noises > -min_noises) ] = np.random.choice(range(-max_noises, -min_noises))
            matrix_noises[s, pick_timesteps] = noises
            
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


class Tempearture_DWD(Data) : 
    def __init__(self, n_stations: int, n_timesteps: int, k: int=5, p_noise_stations: float=0.1, p_noise_timesteps: float=0.1, min_noises=3, max_noises=10) : 
        df = pickle.load(open('data_dvd_reduced.p','rb'))
        df_metadata = pickle.load(open('metadata.p', 'rb'))            
        # idx_stations = np.random.choice(range(len(df_metadata)), size=n_stations, replace=False )
        # idx_rawdata = np.random.choice(range(0, df.shape[1] - n_timesteps))        
        # self._metadata = df_metadata.values[idx_stations]        
        # self._ts_rawdata = self._preprocess_data(df.values[idx_stations, idx_rawdata:idx_rawdata+n_timesteps]   )        
        idx_rawdata = np.random.choice(range(0, df.shape[1] - n_timesteps))        
        print(idx_rawdata)
        self._metadata = df_metadata.values[:n_stations]        
        
        self._ts_rawdata = self._preprocess_data(df.values[:n_stations, idx_rawdata:idx_rawdata+n_timesteps])
        self._k = k          
        super().__init__(p_noise_stations, p_noise_timesteps, min_noises, max_noises)

    def _preprocess_data(self, ts_data: np.ndarray) : 
        df = pd.DataFrame(ts_data)
        df[df<=-40] = np.nan
        df[df>50] = np.nan
        df = df.fillna(method='bfill')
        df = df.fillna(method='ffill')
        return df.values


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


class Executor() : 
    def __init__ (self, data: Data, classifier: Classifier) : 
        self._data = data
        self._classifier = classifier
        

    @property
    def data(self) :
        return self._data
    
    @data.setter
    def strategy(self, data :Data) :
        self._data = data


    @property
    def classifier(self) :
        return self.classifier
    
    @classifier.setter
    def classifier(self, classifier: Classifier) :
        self._classifier = classifier


    def validate(self, station):
        return self._classifier.validate(self._data.ts_data, self._data._neighbor, station)


    def spatial_reference(self,row, ref):
        """
        idx 
            0 : dz 
            1 : latitude
            2 : longitude
        """       

        d_lon = math.radians(float(row[2]) - float(ref[2]))
        d_lat = math.radians(float(row[1]) - float(ref[1]))
        a = math.sin(d_lat / 2) ** 2 + (
            math.cos(math.radians(float(ref[1])))
            * math.cos(math.radians(float(ref[1])))
            * math.sin(d_lon / 2) ** 2
        )
        d = 2.0 * 6371000.0 * math.asin(math.sqrt(a))
        phi, quadrant, orientation = 0.0, 0, 0
        if d > 0.0:
            phi = math.atan2(d_lon, d_lat) * 180.0 / math.pi
            if phi < 0.0:
                phi += 360.0
            quadrant = math.floor(phi / 90) + 1
            orientation = ["N", "E", "S", "W"][int((phi + 45.0) / 90.0) % 4]
        return pd.Series(
            {
                "direction": phi,
                "distance": d,
                "dz": float(row[0]) - float(ref[0]),
                "orientation": orientation,
                "quadrant": quadrant,
            }
        )

    def _regression(X,y) : 
        reg = LinearRegression().fit(X.T,y)
        score = reg.score(X.T,y)
        pred = reg.predict(X.T)
        # print(score)
        # plt.plot(pred,color="red")
        # plt.plot(y)
        # plt.figure()
        # plt.plot(gap)
        return pred

    def _correct_values(pred, y): 
        gap = abs(pred - y)
        idx_for_correction = np.where(gap > np.mean(gap) + 2 * np.std(gap))[0]
        corrected_y = copy.deepcopy(y)
        if len(idx_for_correction) < 1 : 
            print("no more correction possible")
            return y
        else : 
            print("idx for correction", idx_for_correction)
        for i in idx_for_correction : 
            corrected_y[i] = pred[i]
        return corrected_y


    def correct_values(self) :         
        for i in range(len(self._data.ts_data)) :
            pred = self.validate_and_predict(i)            
            
            pred = regression(X,y)
            corrected_y = correct_values(pred, y)
            ts_data[i] = corrected_y
            
    



    def evaluate_validator(self) : 
        RESULT = []
        STAT = {'TP': 0, 'TN':0, 'FP':0, 'FN':0 }
        start = time.time()
        
        for i in range(len(self._data.ts_data)) :
            result = self.validate(i)
            RESULT = RESULT + result
            for r in result : 
                if r in self._data.lst_station_timestep :
                    STAT['TP'] = STAT.get('TP') + 1
                else :
                    STAT['FP'] = STAT.get('FP') + 1
        
        for a in self._data.lst_station_timestep :
            if a not in RESULT:
                STAT['FN'] = STAT.get('FN') + 1
        STAT['runtime'] = round(time.time() - start,4)
        lst_suspcted_stations = set([int(station.split('_')[0]) for station in  RESULT])
        
        toggle_lst_stations = {}

        for station in self._data._picked_stations : 
            toggle_lst_stations[station] = False 
        
        
        lst_false_positive = set()
        for s in lst_suspcted_stations : 
            if s in self._data._picked_stations :
                toggle_lst_stations[s]  = True 
            else : 
                lst_false_positive.add(s)
        STAT['lst_false_positive'] = lst_false_positive
        STAT['toggle_lst_stations'] = toggle_lst_stations


        STAT['n_stations'] = self._data.ts_data.shape[0]
        STAT['n_timestep'] = self._data.ts_data.shape[1]
        STAT['n_tsdata'] = self._data.ts_data.size
        STAT['p_noises'] = len(self._data.lst_station_timestep) / self._data.ts_data.size 
        STAT['TN'] = self._data.ts_data.size - STAT['TP'] - STAT['FP'] - STAT['FN']
        STAT['precision'] = round(STAT['TP'] / (STAT['TP'] + STAT['FP']),3)
        STAT['recall'] = round(STAT['TP'] / (STAT['TP'] + STAT['FN']),3)
        STAT['f1'] = round(2 * STAT['precision'] * STAT['recall'] / (STAT['precision'] + STAT['recall']),3)
        self.result = RESULT        
        return STAT



def regression_based_outlier_detection2(ts_data, neighbor, station, n_regressors, n_variables, eps) :
    result = np.array([], dtype='int')
    coeffs = np.zeros((n_regressors, n_variables + 1))
    score = np.zeros(n_regressors)
    predict = np.zeros((n_regressors, ts_data.shape[1]))
    for i in range(n_regressors) : 
        idx = random.choices(neighbor[station],k=n_variables)
        y = ts_data[station] 
        X = ts_data[idx]
        reg = LinearRegression().fit(X.T, y)
        score[i] = reg.score(X.T, y)
        predict[i] = reg.intercept_ + np.dot(X.T, reg.coef_)
        
    score = score / score.sum()    
    predict = np.dot(predict.T, score)
    original = ts_data[station]
    ix = np.where(abs(predict - original) > eps)
    return [str(station) + '_' + str(x) for x in ix[0]]
    
def regression_based_outlier_detection2(ts_data, neighbor, station, n_regressors, n_variables, eps) :
    result = np.array([], dtype='int')
    coeffs = np.zeros((n_regressors, n_variables + 1))
    score = np.zeros(n_regressors)
    predict = np.zeros((n_regressors, ts_data.shape[1]))
    for i in range(n_regressors) : 
        idx = random.choices(neighbor[station],k=n_variables)
        y = ts_data[station] 
        X = ts_data[idx]
        reg = LinearRegression().fit(X.T, y)
        score[i] = reg.score(X.T, y)
        predict[i] = reg.intercept_ + np.dot(X.T, reg.coef_)
        
    score = score / score.sum()    
    predict = np.dot(predict.T, score)
    original = ts_data[station]
    ix = np.where(abs(predict - original) > eps)
    return [str(station) + '_' + str(x) for x in ix[0]]


def regression_based_outlier_detection3(ts_data, neighbor, station, n_regressors, n_variables, eps) :
    result = np.array([], dtype='int')
    coeffs = np.zeros((n_regressors, n_variables + 1))
    score = np.zeros(n_regressors)
    predict = np.zeros((n_regressors, ts_data.shape[1]))
    for i in range(n_regressors) : 
        idx = random.choices(neighbor[station],k=n_variables)
        y = ts_data[station] 
        X = ts_data[idx]
        reg = LinearRegression().fit(X.T, y)
        score[i] = reg.score(X.T, y)
        predict[i] = reg.intercept_ + np.dot(X.T, reg.coef_)
        
    
    idx = (-score).argsort()[:-1]
    score = score / score.sum()    
    predict = np.dot(predict[idx].T, score[idx])
    original = ts_data[station]
    ix = np.where(abs(predict - original) > eps)
    return [str(station) + '_' + str(x) for x in ix[0]]
    
def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]
    
    
if __name__== '__main__' :
    n_stations = 450
    n_timesteps = 30
    k = 5
    p_noise_stations = 0.1
    p_noise_timesteps = 0.1

    data = Tempearture_DWD(n_stations, n_timesteps, k, p_noise_stations=p_noise_stations, p_noise_timesteps= p_noise_timesteps)
    ensembleregression = EnsembleRegression(n_regressors=5,n_variables=3, eps=2, decision_boundary=0.6, global_search=True)

    executor = Executor(data, ensembleregression)
    print(data._picked_stations)
    # station = np.random.choice(data._picked_stations)
    toggle_stations = {}
    toggle = {}
    total_sum = 0 
    for s in data._picked_stations :                 
        dic = {}
        for t in data.dic_timesteps[s] : 
            dic[t] = False 
            total_sum += 1
        toggle[s] = dic
        toggle_stations[s] = False
    
    eps = 5
    lst_false_positive =[]
    while (True) : 
        ensembleregression = EnsembleRegression(n_regressors=5,n_variables=3, eps=eps, decision_boundary=0.7, global_search=True)
        executor = Executor(data, ensembleregression)
        old_SSE = math.sqrt(np.sum((data.ts_data - data._ts_rawdata) ** 2))
        print("with eps: ", eps)
        print("before correction", old_SSE)
        
        
        for station in range(n_stations) : 
            result = executor._classifier.validate_and_predict(data.ts_data, data._neighbor, station)            
            if result :                                 

                if toggle.get(station) != None : 
                    toggle_stations[station] = True
                    for r in result[0][0] : 
                        toggle[station][r] = True 
                else : 
                    print(station, "is false positive")        
                    lst_false_positive.append([station,result[0]])
                #update
                data.ts_data[station,result[0]] = result[1]
        
        new_SSE = math.sqrt(np.sum((data.ts_data - data._ts_rawdata) ** 2))
        print("after correction", new_SSE)        
        
        if (abs(old_SSE - new_SSE) / old_SSE) < 0.1 : 
            
            break    
    print("false positive", list(lst_false_positive))
    s = 0
    for t in toggle : 
        s += sum(toggle[t].values())
    # print(s, total_sum)
    print("toggle (Positive):", len([v for k,v in toggle_stations.items() if v==True]))
    print("toggle (Negative):", len([v for k,v in toggle_stations.items() if v==False]))
    print("false positive", lst_false_positive)
    

    


# n_stations = 3200
# x_length = 100
# y_length = 100
# time_length = 100
# import time
# if __name__== '__main__' :
#     metadata = create_metadata(n_stations = n_stations, x_length = x_length, y_length = y_length)
#     ts_data = create_temperature_data(n_stations, time_length)
#     neighbor = create_neighor_list(metadata, k=10)
#     ts_data, ANSWER  = add_noise(ts_data, p=0.05)
#     ANSWER.sort()
#     # plt.imshow(ts_data)
#     # plt.show()
#     start = time.time()
#     STAT = evaluator(ts_data, ANSWER, neighbor, n_regressors=10, n_variables=3, eps=0.2)
#     STAT['runtime'] = time.time() - start
#     print(STAT)