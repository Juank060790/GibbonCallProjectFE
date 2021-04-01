'''
Helper functions to perform augmentations.
'''

import numpy as np

SEED = 42 

def blend(audio1: np.ndarray, audio2: np.ndarray, w1: float, w2: float):
    '''
    * Blend two audio based on the weights. 

    * TODO: 
        - Determine the weights
        - Better alternative to blending audio? 
    '''

    return audio1 * w1 + audio2 * w2 

def timeShift(audio: np.ndarray, time: int, sample_rate: int):
    '''
    * Shift an audio segment (wraps around at the end). 
    * Parameters:
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
    '''
    * Augment non-gibbon audio segments.
    * Parameters:
        - augmentation_amount: Number of times to augment non-gibbon call.
        - augmentation_probability: Probability for an audio to be augmented.
        - non_gibbon: Extracted non-gibbon audio segments.
        - sample_rate: Audio sampling rate.
        - alpha: Number of seconds to keep.
    '''
    np.random.seed(SEED)

    augmented_data = []

    for non_gibbon_audio in non_gibbon:
        for i in range(augmentation_amount):
            probability = np.random.uniform() #0-1 output
            if probability <= augmentation_probability:
                #Random time to shift
                #Time starts from 1 instead of 0. This is because shifting 
                #from zero over an entire audio means not shifting at all 
                time = np.random.randint(1, alpha - 1) 
                shifted_audio = timeShift(non_gibbon_audio, time, sample_rate) 
            
                augmented_data.append(shifted_audio)
    
    return np.asarray(augmented_data)

def augmentAudio(augmentation_amount: int, augmentation_probability: float, 
                 gibbon: np.ndarray, non_gibbon: np.ndarray, sample_rate: int,
                 alpha: int):
    '''
    * Augment gibbon calls.
    * Parameters:
        - augmentation_amount: Number of times to augment gibbon segments.
        - augmentation_probability: Probability to augment gibbon segments.
        - gibbon: Extracted gibbon segments.
        - non_gibbon: Extracted non-gibbon segments.
        - sample_rate: Audio sampling rate.
        - alpha: Number of seconds to keep.
    '''
    np.random.seed(SEED)

    augmented_data = []

    #Keep track of the non-gibbon audio file used for blend 
    #and return an array with unused non-gibbon audio 
    index = np.ones(len(non_gibbon - 1), dtype = "bool") 

    for gibbon_audio in gibbon:
        for i in range(augmentation_amount):
            probablity = np.random.uniform() #0-1 output
            if augmentation_probability <= augmentation_probability:
                #Choose a random non-gibbon audio to blend with gibbon audio
                random_non_gibbon = np.random.randint(0, len(non_gibbon) - 1) 
                index[random_non_gibbon] = False 

                time = np.random.randint(1, alpha - 1)
                new_data = timeShift(non_gibbon[random_non_gibbon], time, sample_rate)
                blended_data = blend(gibbon_audio, new_data, 0.9, 0.1)

                augmented_data.append(blended_data)
    
    return np.asarray(augmented_data), non_gibbon[index]
                


