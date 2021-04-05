import numpy as np 

def timeIndex(alpha: int, file_durations: int):
    '''
    * Get the time indices (in seconds) of the total number of segments 
    from a given audio file.
    * Parameters:
        - alpha: Number of seconds to extract.
        - file_durations: Total duration of the file (in seconds).
    '''

    start = []
    end = []

    #If the file only last 10s, then we get at least 1 segment.
    segments = int(file_durations - alpha + 1)

    if (segments < 0):
        raise ValueError(f"Negative segment values: {segments}")
    
    for i in range(segments):
        start.append(i)
        end.append(i + alpha)
    
    return np.asarray(start), np.asarray(end)