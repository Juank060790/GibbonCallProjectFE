'''
Helper functions to load and prepare dataset. 
'''
import numpy as np
import os 
import pathlib 
import pickle 
import tensorflow as tf 
from sklearn.model_selection import train_test_split 

FOLDER = "extracted_audio"

def loadSpectrogram(folder: str):
    '''
    Load arrays of gibbon and non-gibbon spectrogram from a given folder, prepare 
    a numpy array for labels, and apply train_test_split.
    '''

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
    
    gibbon = np.asarray(gibbon, dtype = "float32")
    non_gibbon = np.asarray(non_gibbon, dtype = "float32")
        
    #Sample so that the two classes data points equal
    sample_amount = gibbon.shape[0]
    non_gibbon = non_gibbon[np.random.choice(gibbon.shape[0], sample_amount, replace = True)]

    #Labels 
    y_true = np.ones(len(gibbon), dtype = "float32")
    y_false = np.zeros(len(non_gibbon), dtype = "float32")

    #Validation split 
    X = np.concatenate([gibbon, non_gibbon])
    y = tf.keras.utils.to_categorical(np.concatenate([y_true, y_false]))

    del gibbon, non_gibbon 

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    del X, y 

    return x_train, y_train, x_test, y_test

def loadDataset(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, 
                y_test: np.ndarray):
    '''
    Load a train and validation tensorflow dataset. 
    '''
    train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    validation = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    return train, validation  
    
# if __name__ == "__main__":
#     x_train, y_train, x_test, y_train = loadSpectrogram(FOLDER)
#     ds = loadDataset(x_train, y_train, x_test, y_test)
