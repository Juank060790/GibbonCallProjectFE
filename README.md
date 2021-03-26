* Last updated: 03/26/2021
* Changes: 
     - extract.py: Added helper functions
     - **TODO**: 
          1) Determine approach to extracting data, i.e hop length, what 
          data to keep...
          2) Run data extraction and see how many gibbon segments we have. 
* Logs:
     - 03/25/2021: Helper function to extract path and audio segments 
     (see helper/extract.py)
     - 03/26/2021: Analzyed data (see EDA.ipynb). 
* NOTE: 
     - The **BIG** problem with our dataset is that we're assuming input 
     segments only contain gibbon, bird + insects, individually. In reality, 
     audio segments may overlapped. 
     - Solutions:
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
|--- _helper 
     |--- extract.py: Helper functions to extract raw audio files. 
|--- _Raw Audio Files
     |--- **\*.WAV, raw audio files.
|--- _Selection Tables
     |--- *.txt, labelled segments in an audio.
|--- README.md
|--- extraction.ipynb: Tests extracting audio files 
|--- EDA.ipynb: Analysis of data. 
```