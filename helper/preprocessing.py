'''
Main driver to extract audio and perform augmentation.

NOTE: Assumes that in a given label .txt, the audio path is the same 
(has been verified).
'''

from extract import * #Helper functions for audio extraction
from augmentation import * #Helper functions for augmentation

import os 
import pathlib 
import pickle 
import librosa 
import time 

LABEL = "Selection Tables"
OUTPUT = "extracted_audio"

def preprocess(output_path : str, label_path: str, sample_rate: int, alpha: int,
               jump_seconds: int):
    '''
    * Purpose: Intermediate function that calls readLabels() and extractAudio(),
               found in extract.py. The extracted segments are stored into a 
               pkl file. 
    * Parameters:
        - output_path: Output folder to store extracted audio segments. 
        - label_path: Path to label files.
        - sample_rate: The rate to sample an audio.
        - alpha: Number of seconds in extracted audio segments.
        - jump_seconds: Slicing window hop size.
    '''

    df = readLabels(label_path, sample_rate)
    filename = df["Path"][0]
    audio_pkl = label_path.split("\\")[-1] + ".pkl"
    # print(audio_pkl)

    audio, _ = librosa.load(filename, sr = sample_rate)

    gibbon, non_gibbon = extractAudio(df, audio, sample_rate, alpha, jump_seconds)

    del audio 

    absolute_path = os.path.join(os.getcwd(), output_path)

    if len(gibbon != 0):
        with open(os.path.join(absolute_path, "gibbon", audio_pkl), "wb") as file:
            pickle.dump(gibbon, file)
    
    if len(non_gibbon != 0):
        with open(os.path.join(absolute_path, "non_gibbon",  audio_pkl), "wb") as file:
            pickle.dump(non_gibbon, file)   
    
    toSpectrogram(output_path, audio_pkl, gibbon, non_gibbon, sample_rate)
    
def augment(filename: str, gibbon: np.ndarray, non_gibbon: np.ndarray, alpha: int, 
                        sample_rate: int, augmentation_amount_noise: int, 
                        augmentation_probability: float, 
                        augmentation_amount_gibbon: int):
    '''
    * Augment extracted audio.
    * Parameters:
        - gibbon: Extracted gibbon call. 
        - non_gibbon: Extracted non-gibbon call. 
        - alpha: Number of seconds to keep.
        - sample_rate: Audio sampling rate.
        - augmentation_amount_noise: Number of times to augment non-gibbon call. 
        - augmentation_probability: Probability for an audio segment to be augmented.
        - augmentation_amount_gibbon: Number of times to augment gibbon call.
    '''

    non_gibbon_augmented = augmentBackground(
        augmentation_amount_noise, augmentation_probability, non_gibbon, 
        sample_rate, alpha
    )
    gibbon_augmented, unusued_non_gibbon_augmented = augmentAudio(
        augmentation_amount_gibbon, augmentation_probability, gibbon, 
        non_gibbon_augmented, sample_rate, alpha 
    )

    # print(
    # f"""
    # Initial non-gibbon calls {non_gibbon_augmented.shape}, \
    # Current non-gibbon calls {unusued_non_gibbon_augmented.shape}
    # Gibbon calls {gibbon_augmented.shape}
    # """)

    return gibbon_augmented, non_gibbon_augmented

def toSpectrogram(output_path: str, filename: str, gibbon: np.ndarray, 
                  non_gibbon: np.ndarray, sample_rate: int):
    '''
    * Convert audio to spectrogram and save to .pkl file.
    * Parameters: 
        - output_path: Output folder to store extracted audio spectrogram. 
        - filename: Segment's audio name. 
        - gibbon: Extracted gibbon audio. 
        - non_gibbon: Extracted non-gibbon audio.
    '''

    if len(gibbon) != 0:
        # print(f"Gibbon melspectrogram shape: {gibbon.shape}")
        absolute_path = os.path.join(os.getcwd(), output_path)
        gibbon_spectrogram = extractSpectrogram(gibbon, sample_rate)

        with open(os.path.join(absolute_path, "gibbon_spectrogram", filename), "wb") as file:
            pickle.dump(gibbon, file)

        try: 
            non_gibbon = non_gibbon[np.random.choice(non_gibbon.shape[0],
                                gibbon.shape[0], replace = True)]
            # print(f"Non-gibbon melspectrogram shape: {non_gibbon.shape}")
            non_gibbon_spectrogram = extractSpectrogram(non_gibbon, sample_rate)
            with open(os.path.join(absolute_path, "non_gibbon_spectrogram", filename), "wb") as file:
                pickle.dump(non_gibbon, file)  
        except Exception:
            pass 


def main(output_path: str, label_path: str):
    '''
    * Purpose: Main driver to extract segments from raw audio data.
    * Parameters:
        - output_path: Output folder to store extracted audio segments. 
        - label_path: Path to label files.
    '''
    
    print("Begin extracting raw audio files")
    start = time.time()

    sample_rate = 4800
    alpha = 10
    jump_seconds = 1

    number_iterations = 1
    augmentation_probability = 1
    augmentation_amount_noise = 2
    augmentation_amount_gibbon = 10 

    files = [str(file) for file in 
             pathlib.Path(os.path.join(os.getcwd(), label_path)).glob("*.txt")]
    print(f"Number of files {len(files)}")

    for file in files:
        try:
            preprocess(
            output_path, file, sample_rate, alpha, jump_seconds
            )
        except:
            print(f"{file} doesn't exist")
    
    # gibbon_file = [str(file) for file in pathlib.Path(os.path.join(output_path, "gibbon")).glob("*")]
    # non_gibbon_file = [str(file) for file in pathlib.Path(os.path.join(output_path, "non_gibbon")).glob("*")]

    end = time.time()
    print(f"Time elapsed: {end - start}")

if __name__ == "__main__":
    main(OUTPUT, LABEL)
