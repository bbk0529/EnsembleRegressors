from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from new_predictor_util import create_idx_stations_w_noises

def plot_coefficient(data, station, k, lst_corrupted_stations) : 
    ts_data = data.ts_data
    neighbor = data._neighbor
    X = ts_data[neighbor[station]]
    y = ts_data[station]
    
    idx_stations_w_noises = create_idx_stations_w_noises(lst_corrupted_stations, neighbor, station)

    reg = LinearRegression().fit(X.T, y)    
    
    fig, ax = plt.subplots(figsize=(20,3))
    p1 = ax.bar(range(k), reg.coef_, label='coeff')
    p2 = ax.bar(range(k),idx_stations_w_noises * - max(abs(reg.coef_)), alpha=0.3, color='red', label='flag')

    ax.axhline(0, color='grey', linewidth=0.8)
    ax.set_ylabel('coeff')
    ax.set_title('Coefficients of Linear regression')
    ax.set_xticks(range(k))
    ax.set_xticklabels((neighbor[station]))
    ax.legend()
    plt.show()    

def plot_indicators(data, station, pred_by_new, answer, smoothpredictor) : 
    ts_data =data.ts_data
    neighbor = data._neighbor
    corrected_y = smoothpredictor.correct(ts_data[neighbor[station]], ts_data[station],graph=False)

    X = ts_data[neighbor[station]]
    y = ts_data[station]
    y_max = np.max(X)
    y_min = np.min(X)

    plt.figure(figsize=(20,5))
    
    reg = LinearRegression().fit(X.T, y)
    plt.plot(ts_data[station], color='blue', linewidth=3)
    plt.plot(reg.predict(X.T), color='red', linewidth=3, alpha=0.8, linestyle='--')
    plt.plot(ts_data[neighbor[station]].T, color='grey', alpha=0.6)
    plt.legend(['data_w_noise','predicted_data','neighbors_data'])
    plt.ylim([y_min,y_max])



    plt.figure(figsize=(20,5))
    plt.title(station)    
    plt.plot(ts_data[station], color='blue', linewidth=2)
    plt.plot(data.ts_rawdata[station], color='green', linewidth=3, linestyle='--', alpha=0.8)    
    plt.plot(ts_data[neighbor[station]].T, color='grey', linewidth=1)
    plt.legend(['data_w_noises', 'data_wo_noises', 'neighbors',])
    plt.ylim([y_min,y_max])

    plt.figure(figsize=(20,5)) 
    plt.plot(data.ts_rawdata[station], color='green', linewidth=2, linestyle='--')    
    plt.plot(ts_data[station], color='blue')
    plt.plot(corrected_y, color='red',  linewidth=3, linestyle='--', alpha=0.6)
    plt.legend(['data_wo_noises', 'data_w_noise', 'corrected_data',])
    plt.ylim([y_min,y_max])

    #plotting indicators
    plt.figure(figsize=(20,1)) 
    bar1 = np.zeros(len(ts_data[station]))
    bar2 = np.zeros(len(ts_data[station]))
    
    y_max = max(abs((ts_data[station])))
    bar1[answer] = y_max
    bar2[pred_by_new] = y_max


    
    plt.bar(np.arange(len(bar1)), bar1, color='red', alpha=0.8, label='answer')
    plt.bar(np.arange(len(bar2)), -bar2, color='green', alpha=0.8, label='prediction')    
    plt.legend(['answer','prediction'])


def plot_noises(data) : 
    noises = data.matrix_noises.flatten()[abs(data.matrix_noises.flatten()) > 0]
    fig, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(21,7))
    ax1.imshow(data.matrix_noises)
    ax2.imshow(data.matrix_noises[:100,:100])
    ax3.hist(noises)