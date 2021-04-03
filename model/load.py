'''
Helper functions to load and prepare dataset. 
'''
import numpy as np
import os 
import pathlib 
import pickle 

FOLDER = "extracted_audio"

def loadSpectrogram(folder: str):
    gibbon_path = os.path.join(os.getcwd(), folder, "gibbon_spectrogram")
    gibbon_spectrogram = [str(file) for file in pathlib.Path(gibbon_path).glob("*.pkl")]

    non_gibbon_path = os.path.join(os.getcwd(), folder, "non_gibbon_spectrogram")
    non_gibbon_spectrogram = [str(file) for file in pathlib.Path(non_gibbon_path).glob("*.pkl")]

    gibbon, non_gibbon = [], []

    for file in gibbon_spectrogram:
        with open(file, "rb") as f:
            gibbon.extend(pickle.load(f))
    
    for file in non_gibbon_spectrogram:
        with open(file, "rb") as f:
            non_gibbon.extend(pickle.load(f))
    
    gibbon = np.asarray(gibbon)
    non_gibbon = np.asarray(non_gibbon)
    
    return gibbon, non_gibbon

if __name__ == "__main__":
    gibbon, non_gibbon = loadSpectrogram(FOLDER)
    print(gibbon.shape)
    print(non_gibbon.shape)