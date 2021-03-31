'''
Helper functions to perform augmentations.
'''

import numpy as np

SEED = 42 

def blend(audio1: np.ndarray, audio2: np.ndarray, w1: float, w2: float):
    '''
    Blend two audio based on the weights. 

    TODO: 
        - Determine the weights
        - Better alternative to blending audio? 
    '''

    return gibbon * w1 + gibbon * w2 

def timeShift(audio: np.ndarray, time: int, sample_rate: int):
    '''
    Shift an audio segment (wraps around at the end). 
    Parameters:
        - audio: Extracted audio segment.
        - time: Number of seconds to shift.
        - sample_rate: Audio sampling rate.
    '''
    timestamp = sample_rate * time

    shifted_audio = np.zeros(len(audio))
    shifted_audio[:timestamp] = audio[-1 * timestamp:]
    shifted_audio[timestamp:] = audio[ :-1 * timestamp]

    return shifted_audio

def augmentBackground(augmentation_amount: int, augmentation_probability: float, 
                      non_gibbon: np.ndarray, sample_rate: int, alpha: int):

    np.random.seed(SEED)

    augmented_data = []

    for data in non_gibbon:
        for i in range(augmentation_amount):
            probability = np.random.uniform() #0-1 output
            if probability <= augmentation_probability:
                #Random time to shift
                #Time starts from 1 instead of 0. This is because shifting 
                #from zero over an entire audio means not shifting at all 
                time = np.random.randint(1, alpha - 1) 
                shifted_audio = timeShift(non_gibbon, time, sample_rate) 
            
                augmented_data.append(shifted_audio)
    
    return np.asarray(augmented_data)

def augmentAudio(augmentation_amount: int, augmentation_probability: float, 
                 gibbon: np.ndarray, non_gibbon: np.ndarray, sample_rate: int,
                 alpha: int):
    
    np.random.seed(SEED)

    augmented_data = []

    for data in gibbon:
        for i in range(augmentation_amount):
            probablity = np.random.uniform() #0-1 output
            if probability <= augmentation_probability:
                
                background = np.random.randint(0, len(non_gibbon) - 1)
                time = np.random.randint(1, alpha - 1)
                    
                


