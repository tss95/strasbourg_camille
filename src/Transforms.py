import matplotlib.pyplot as plt
from scipy.signal import detrend, butter, filtfilt
import numpy as np
import random



class AddNoise:
    def __init__(self, noise_level):
        self.noise_level = noise_level
        
    def __name__(self):
        return "AddNoise"
    
    def __call__(self, data):
        noise = np.random.normal(0, self.noise_level, data.shape)
        return data + noise

class LocalMinMaxNorm:
    
    def __name__(self):
        return "LocalMinMaxNorm"
    
    def __call__(self, data):
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val)


    
    


