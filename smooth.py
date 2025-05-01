import numpy as np

def smooth(data, window_size=100):

    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')