import pathlib 
import os
import regex as re
import librosa
import pandas as pd 
import numpy as np

def readLabels(path: str, sample_rate: int):
    '''
    * Purpose: Given a path to a text file containing the labels, read in the 
    information and convert it to a dataframe. 
    * Parameters: 
        path: Path to file containing the labels. 
        sample_rate: Audio sampling rate. 
    * Returns:  
        df: pd.DataFrame. Columns are start and end timestamps converted in 
        sample rate, path to audio, and the label.
    '''

    def convertPath(path: str):
        '''
        * Purpose: Convert a path to absolute path.
        '''
        relative_path = re.findall("Raw Audio Files.+", path)[0]
        return os.path.join(os.getcwd(), re.sub("\((.)*\)", \
                        "(September to October 2020)", relative_path))

    df = pd.read_csv(path, delimiter = "\t")
    df.drop(columns = ["Selection", "View", "Channel", "Begin File"], 
            inplace = True)
    df.columns = ["Start", "End", "Low", "Height", "Path", "Label"]

    df["Path"] = df["Path"].apply(convertPath)
    df["Start"] = df["Start"] * sample_rate
    df["End"] = df["End"] * sample_rate

    return df 

def extractAudio(df: pd.DataFrame, audio: np.ndarray, sample_rate: int, 
                 alpha: int = 10, jump_seconds: int = 1):
    '''
    * Purpose: Extract calls using a sliding window approach.
    * Parameters:
        - df: Contains timestamps and labels.
        - audio: Librosa loaded audio
        - sample_rate: Audio sampling rate. 
        - alpha: How many seconds to slide over a given timestamp. 
        - jump_seconds: Hops between sliding window. 
    * Returns: 
        - An array containing gibbon segments and an array containing 
        non-gibbon segments. 
    * TODO:
        - Determine how we want to deal with non-gibbon class and extract 
        accordingly.
    '''


    positive = ["gc", "mm", "sc"]
    alpha_converted = alpha * sample_rate 
    gibbon, non_gibbon = [], []

    for _, row in df.iterrows():
        jump = 0
        while True:
            start_position = row["Start"] - sample_rate - \
                             (jump * jump_seconds * sample_rate)
            end_position = start_position + alpha_converted 
            jump += 1

            if end_position <= row["End"]:
                break 

            extract_segment = audio[int(start_position): int(end_position)]

            if row["Label"] in positive:
                gibbon.append(extract_segment)
            else:
                non_gibbon.append(extract_segment)
        
    return np.asarray(gibbon, dtype = "object"), \
        np.asarray(non_gibbon, dtype = "object")


