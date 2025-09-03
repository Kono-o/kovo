import sounddevice as sd
import threading
import numpy as np
from .state import State

from pydbus import SessionBus

def songfetch():
    bus = SessionBus()
    dbus_object = bus.get('org.freedesktop.DBus', '/')
    all_names = dbus_object.ListNames()
    players = [name for name in all_names if name.startswith("org.mpris.MediaPlayer2.")]
    if not players:
        return
    for player_name in players:
        try:
            player = bus.get(player_name, "/org/mpris/MediaPlayer2")
            metadata = player.Metadata
            artist_list = metadata.get('xesam:artist', [])
            State.artist = artist_list[0] if artist_list else None
            State.title = metadata.get('xesam:title',  None)
            State.album = metadata.get('xesam:album',  None)
        except Exception:
            continue
    
def audiofetch():
    def_frame_count = 1024
    def_device = 'pulse'
    def_sample_rate = 44100
    def callback(indata, frames, time_info, status):
        try:
            # Store raw stereo PCM data
            State.L = indata[:, 0]  # Left raw int32
            State.R = indata[:, 1]  # Right raw int32
            
            # Normalize to [-1, 1] for FFT processing
            State.Ln = State.L.astype(np.float64) / 2147483648.0
            State.Rn = State.R.astype(np.float64) / 2147483648.0
            
            # Calculate RMS volumes (perceived loudness)
            State.Lvol = np.sqrt(np.mean(State.Ln ** 2))
            State.Rvol = np.sqrt(np.mean(State.Rn ** 2))
            State.Mvol = np.sqrt(np.mean(((State.Ln + State.Rn) * 0.5) ** 2))
            
            # Calculate peak amplitudes
            State.Lpeak = np.max(np.abs(State.Ln))
            State.Rpeak = np.max(np.abs(State.Rn))
            State.Mpeak = max(State.Lpeak, State.Rpeak)
            
            # Audio info
            State.sample_rate = int(sd.query_devices(sd.default.device[0], 'input')['default_samplerate'])
            State.frame_count = def_frame_count
            State.is_silent = State.Mvol < 0.001  # -60dB threshold
            State.is_up = True
            songfetch()
            
        except Exception as e:
            State.is_up = False
            print(f"[audiofetch] callback error: {e}")
    
    try:
        stream = sd.InputStream(
            device=def_device,
            channels=2,
            samplerate=def_sample_rate,
            blocksize=def_frame_count,
            dtype='int32',
            callback=callback,
        )
        def run():
            with stream:
                threading.Event().wait()
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
    except Exception as e:
        print(f"[audiofetch] setup failed: {e}")
        State.is_up = False