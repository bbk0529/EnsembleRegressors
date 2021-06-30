
import pandas as pd
import math
import numpy as np


def pick_FS_station(executorNew, executorOld, neighbor, idx_stations_w_noises, dic_timesteps) : 
    while True : 
        station = np.random.choice(range(len(neighbor)))
        answer, pred_by_new, pred_by_old  = validate_for_comparision(executorNew, executorOld, neighbor, station, idx_stations_w_noises, dic_timesteps)
        if (len(answer) == 0) and len(pred_by_old) > 0 : 
            break
    print("Station: {} \t w. corrupted neighbors: {} ".format(station, neighbor[station][idx_stations_w_noises]))
    print("\t Answer:          : {}, \t {}".format(0, answer))
    print("\t Predicted by new : {}, \t {}".format(levenshteinDistance(answer, pred_by_new), pred_by_new))
    print("\t Predicted by old : {}, \t {}".format(levenshteinDistance(answer, pred_by_old), pred_by_old))
    return station

def validate_for_comparision(executorNew, executorOld, neighbor, station, idx_stations_w_noises, dic_timesteps, output=False) :     
    try : 
        answer = sorted(dic_timesteps[station])
    except : 
        answer = []

    pred_by_new = sorted([int(x.split('_')[1]) for x in executorNew.validate(station)])
    pred_by_old = sorted([int(x.split('_')[1]) for x in executorOld.validate(station)])
    if output : 
        print("Station: {} \t w. corrupted neighbors: {} ".format(station, neighbor[station][idx_stations_w_noises]))
        print("\t Answer:          : {}, \t {}".format(0, answer))
        print("\t Predicted by new : {}, \t {}".format(levenshteinDistance(answer, pred_by_new), pred_by_new))
        print("\t Predicted by old : {}, \t {}".format(levenshteinDistance(answer, pred_by_old), pred_by_old))

    return answer, pred_by_new, pred_by_old

def pick_station_randomly(neighbor, lst_corrupted_stations, w_noise = False) : 

    if w_noise : 
        while True : 
            station = np.random.choice(lst_corrupted_stations)
            
            neighbor_w_stations = [s for s in neighbor[station] if s in lst_corrupted_stations]
            if len(neighbor_w_stations) > 0 : 
                break
            
            
            
    else : 
        station = np.random.choice(lst_corrupted_stations)
    
    idx_stations_w_noises = create_idx_stations_w_noises(lst_corrupted_stations, neighbor, station)

    return station, idx_stations_w_noises


def create_idx_stations_w_noises(lst_corrupted_stations, neighbor, station) : 
    idx_stations_w_noises = (neighbor[station] >= np.inf )
    for s in neighbor[station] : 
        if s in lst_corrupted_stations : 
            idx_stations_w_noises = idx_stations_w_noises | (neighbor[station] == s)        

    return idx_stations_w_noises 

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
    
def spatial_reference(row, ref):
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


def overall_evaluation(executorNew, executorOld) : 
    resultNew = executorNew.evaluate_validator()
    print("="*50)
    print("Type\tPrec\t Recall\t F1\t runtime")
    print("="*50)
    print(" New\t {}\t  {}\t  {}\t  {}\t".format(resultNew['precision'], resultNew['recall'], resultNew['f1'], resultNew['runtime']))
    resultOld = executorOld.evaluate_validator()
    print(" Old\t {}\t  {}\t  {}\t  {}\t".format(resultOld['precision'], resultOld['recall'], resultOld['f1'], resultOld['runtime']))