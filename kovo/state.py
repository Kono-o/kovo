import time

class State:
    # TUI State
    frame = 0                  # frames rendered in the terminal
    fps = 0                    # current fps
    target_fps = 60            # desired fps
    is_up = False              # singleton for starting the loop
    _last_time = time.time()   # Internal timing (float) - used for FPS calculation

    # Raw audio channels (int32 PCM from PulseAudio)
    L = None                   # Left channel raw (numpy array, shape: (1024,), dtype: int32)
    R = None                   # Right channel raw (numpy array, shape: (1024,), dtype: int32)
    
    # Normalized audio channels (for FFT processing)
    Ln = None                  # Left normalized (numpy array, shape: (1024,), dtype: float64, range: [-1,1])
    Rn = None                  # Right normalized (numpy array, shape: (1024,), dtype: float64, range: [-1,1])
    
    # Audio info
    sample_rate = 44100        # Sample rate (int, Hz)
    frame_count = 0            # Frames in buffer (int, should be 1024)
    is_silent = True           # Silence detection (bool)
    artist = None
    title = None 
    album = None
    
    # Volume metrics  
    Lvol = 0.0                 # Left RMS power (float, range: [0,1])
    Rvol = 0.0                 # Right RMS power (float, range: [0,1])
    Mvol = 0.0                 # Mono RMS power (float, range: [0,1]) - overall volume
    Lpeak = 0.0                # Left peak amplitude (float, range: [0,1])
    Rpeak = 0.0                # Right peak amplitude (float, range: [0,1])
    Mpeak = 0.0                # Mono peak amplitude (float, range: [0,1])
    