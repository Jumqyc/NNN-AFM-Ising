import numpy as np
import statsmodels.api as sm
from scipy.stats import ks_2samp
def is_Equilibrium(data: np.array, 
                     auto_corr_thresh: float = 1e-2,
                     stat_threshold: float = 0.1,
                     p_threshold: float = 0.01,
                     min_data_points: int = 200) -> bool:
    
    
    # auto-corr analysis
    def acf(vec):
        nlags = int(len(vec) * 3/4)
        if nlags < 1:
            return True, -1
        
        try:
            acf = sm.tsa.acf(vec, nlags=nlags, fft=False)
        except:  
            return True, -1
            
        acf_abs = np.abs(acf)
        lag_candidates = np.where(acf_abs <= auto_corr_thresh)[0]
        
        if len(lag_candidates) == 0:
            return False, -1  # No possible lag
        return False, lag_candidates[0]  # return false
    
    same, lag = acf(data)
    if same or lag < 1:
        return False
    
    # slicing
    sampled_data = data[::lag]
    
    # Make sure there is enough sample
    if len(sampled_data) < 2:
        return False
    
    # separate into two parts
    split_idx = len(sampled_data) // 2
    part1 = sampled_data[:split_idx]
    part2 = sampled_data[split_idx:]
    
    # KS sampling
    stat, p_value = ks_2samp(part1, part2)
    
    # returns bool
    return (
        (p_value >= p_threshold or stat <= stat_threshold) 
        and len(sampled_data) >= min_data_points
    )