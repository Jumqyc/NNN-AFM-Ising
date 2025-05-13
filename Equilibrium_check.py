import numpy as np
from statsmodels.tsa.stattools import acf as stacf
from scipy.stats import ks_2samp

def is_Equilibrium(data: np.array, 
                     acorr_threshold: float = 0.1,
                     stat_threshold: float = 0.1,
                     p_threshold: float = 0.01,
                     min_data_points: int = 50) -> bool:
    # auto-corr analysis
    nlags = len(data)

    #normally min-data-points = 500 
    #saves time since the program quits in the first part and do not do any time-consuming calculation
        
    acf = stacf(data, nlags=nlags, fft=False)
    acf_abs = np.abs(acf)
    lag_candidates = np.where(acf_abs <= acorr_threshold)[0]
    if len(lag_candidates) == 0:
        return False
    lag = lag_candidates[0]
    if lag < 1:
        return False
    # slicing
    sampled_data = data[::lag]
    
    # Make sure there is enough sample
    if len(sampled_data) < min_data_points:
        return False
    
    # separate into two parts
    split_idx = len(sampled_data) // 2
    part1 = sampled_data[:split_idx]
    part2 = sampled_data[split_idx:]
    
    # KS sampling
    stat, p_value = ks_2samp(part1, part2)
    
    # returns bool
    return p_value >= p_threshold or stat <= stat_threshold
