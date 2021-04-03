* Last updated: 03/30/2021
* **TODO**: 
     1) Determine approach to extracting data, i.e hop length, what 
     data to keep...
     3) **BUG**: In extract.py, extractAudio(), there are instances of 
                    shape mismatched. We expect the length of the extracted 
                    audio segment to be 48,000; this is the sampling_rate 
                    multiply by number of seconds to keep.
     4) Determine appropriate parameters to convert an array into 
          melspectrogram.   
     5) Data augmentation. 

* Logs:
     - 03/25/2021: Helper function to extract path and audio segments 
     (see helper/extract.py).
     - 03/26/2021: Analzyed data (see EDA.ipynb).
     - 03/27/2021: Analyzed research paper's label (see EDA.ipynb).
          - **Findings**: They only cared about the longest duration of the 
          calling bout. 
     - 03/30/2021: Helper function to extract audio segments and store 
                   into .pkl files. 
     - 04/01/2021: Helper function to convert extracted audio segments into 
                   melspectrogram and store into .pkl files.
     - 04/03/2021: Build a baseline model. 
          
* NOTE: 
     - The **BIG** problem with our dataset is that we're assuming input 
     segments only contain gibbon, bird + insects, individually. In reality, 
     audio segments may overlapped. 
     - Research paper dealt with this problem by blending non-gibbon audio 
     and gibbon audio.
     - Other solutions:
          1) Sound synthesis (could be risky).
          2) Autoencoder (detect the nature of gibbon call).
          3) Independent component analysis 
          https://www.youtube.com/watch?v=T0HP9cxri0A
          - The idea would be to run this algorithm (possibly model) before 
          prediction and feed separated audio frequency into our prediction 
          model. 
---
**Label Keys**:
* **TODO**: Include a spectrogram sample for each type of call.
* **NOTE**: *boom* and *coda* calls are not labelled. 
* sc: staccato 
* mm: multi-modulated
* coda: Special type of *mm* call which the male only makes in response to the female great call
* gc: great call. These are made by female gibbons. Immatures may also join in but will be quieter than the adult female(s). It consists of a series of very rapid calls that build in frequency and loudness. The male coda call comes in at the end.

**Directory Path**:
```
.
|--- _extracted_audio
     |--- _gibbon: Gibbon audio segments extracted from raw data.
     |--- _non_gibbon: Non-gibbon audio segments extracted from raw data. 
|--- _helper 
     |--- augmentation.py: Helper functions to augment audio files.
     |--- extract.py: Helper functions to extract raw audio files. 
     |--- preprocessing.py: Main driver to extract and augment audio files.
|--- _model
     |--- baseline.py: Baseline model.
     |--- load.py: Load and extract melspectrogram.
|--- _prediction
     |--- post_processing.py
|--- _Raw Audio Files
     |--- **\*.WAV, raw audio files.
|--- _research_paper
     |--- Train_Labels: Research paper's labelling
     |--- research.pdf
|--- _Selection Tables
     |--- *.txt, labelled segments in an audio.
|--- _test_files
     |--- *.WAV, audio files set aside for testing. 
|--- README.md
|--- extraction.ipynb: Tests extracting audio files 
|--- EDA.ipynb: Analysis of data. 
|--- requirements.txt: Dependencies. 
|--- Untitled.ipynb
```
