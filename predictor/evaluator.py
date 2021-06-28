from abc import abstractmethod, ABC
class Evaluator(ABC) :
    pass 


class ConnfusionMatrix(Evaluator) : 
    pass
    """
    
    F1 SCORE



    """


class LevineDistance(Evaluator):
    pass


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