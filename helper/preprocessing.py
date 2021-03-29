'''
Last updated: 03/27/2021

Main driver to extract audio and perform augmentation.

NOTE: Assumes that in a given label .txt, the audio path is the same 
(has been verified).
'''

from extract import *

import os 
import pathlib 
import pickle 
import librosa 

LABEL = "Selection Tables"
OUTPUT = "extracted_audio"

def preprocess(output_path : str, label_path: str, sample_rate: int, alpha: int):
    '''
    * Purpose: Intermediate function that calls readLabels() and extractAudio(),
               found in extract.py. The extracted segments are stored into a 
               pkl file. 
    * Parameters:
        output_path: Output folder to store extracted audio segments. 
        label_path: Path to label files.
        sample_rate: The rate to sample an audio.
        alpha: Number of seconds in extracted audio segments.
    '''

    df = readLabels(label_path, sample_rate)
    filename = df["Path"][0]
    audio_pkl = filename.split("\\")[-1][:filename.find("WAV")] + ".pkl"

    audio, _ = librosa.load(filename, sr = sample_rate)
    gibbon, non_gibbon = extractAudio(df, audio, sample_rate, alpha, jump_seconds)

    absolute_path = os.path.join(os.getcwd(), output_path)

    with open(os.path.join(absolute_path, "gibbon", audio_pkl), "wb") as file:
        pickle.dump(gibbon, file)

    with open(os.path.join(absolute_path, "non_gibbon",  audio_pkl), "wb") as file:
        pickle.dump(non_gibbon, file)   

    return gibbon, non_gibbon 

def main(output_path: str, label_path: str):
    '''
    Purpose: Main driver to extract segments from raw audio data.
    Parameters:
        output_path: Output folder to store extracted audio segments. 
        label_path: Path to label files.
    Returns: 
        None
    '''
    
    sample_rate = 4800
    alpha = 10
    jump_seconds = 1
    background_noise = 2
    gibbon_augmentation = 10
    probability = 1.0

    files = [str(file) for file in 
             pathlib.Path(os.path.join(os.getcwd(), label_path)).glob("*.txt")]
    
    for file in files:
        gibbon, non_gibbon = preprocess(output_path, file, sample_rate, alpha)

        break 

if __name__ == "__main__":
    main(LABEL, OUTPUT)
