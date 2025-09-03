from prompt_toolkit.formatted_text import to_formatted_text
from .state import State

# --- COLOR FORMATTING ---
def color(fg_hex, bg_hex=None, bold=False, italic=False):
    parts = []
    if bg_hex:
        parts.append(f'bg:{bg_hex}')
    parts.append(fg_hex)
    if bold:
        parts.append('bold')
    if italic:
        parts.append('italic')
    return ' '.join(parts)

def darken(hexcode: str, factor: float = 0.2) -> str:
    hexcode = hexcode.lstrip("#")
    if len(hexcode) != 6:
        raise ValueError("Hex code must be in format #RRGGBB")

    r = int(hexcode[0:2], 16)
    g = int(hexcode[2:4], 16)
    b = int(hexcode[4:6], 16)

    r = int(r * factor)
    g = int(g * factor)
    b = int(b * factor)
    return "#{:02x}{:02x}{:02x}".format(r, g, b)

# --- COLOR CONSTANTS ---
S0  = "#E5E5FF"
S1  = "#DBDBFF"
S2  = "#D1D1FF"
S3  = "#C6C6FF"
S4  = "#BBBBFF"
S5  = "#AFAEFF"
S6  = "#A3A3FF"
S7  = "#9998FF"
S8  = "#8F8DFF"
S9  = "#8782FF"
S10 = "#7F77FF"
S11 = "#796CFF"
S12 = "#7460FF"
S13 = "#7053FF"
S14 = "#6E4AFF"
S15 = "#6D42FF"
S16 = "#7140FF"
S17 = "#7542FF"

from dataclasses import dataclass

@dataclass(frozen=True)
class ColorVariants:
    fg: str     # normal fg
    ital: str   # italic
    b_fg: str   # bold fg
    bg: str     # fg on bg
    b_bg: str   # bold fg on bg
    d_fg: str   # darkened fg
    d_b_fg: str # darkened bold fg
    d_bg: str   # darkened fg on bg
    d_b_bg: str # darkened bold fg on bg

def make_variants(hexcode: str, name: str, darken_factor: float = 0.5) -> ColorVariants:
    d_hex = darken(hexcode, darken_factor)
    return ColorVariants(
        fg    = color(hexcode),
        ital    =color(hexcode, italic=True),
        b_fg  = color(hexcode, bold=True),
        bg    = color(BLACK, hexcode),
        b_bg  = color(BLACK, hexcode, bold=True),
        d_fg    = color(d_hex),
        d_b_fg  = color(d_hex, bold=True),
        d_bg    = color(BLACK, d_hex),
        d_b_bg  = color(BLACK, d_hex, bold=True),
    )

WHITE   = "#FFFFFF"
BLACK   = "#000000"

palette = {
    "white":   make_variants(WHITE, "white"),
    "black":   make_variants(BLACK, "black"),
}

# fg
s0  = color(S0)
s1  = color(S1)
s2  = color(S2)
s3  = color(S3)
s4  = color(S4)
s5  = color(S5)
s6  = color(S6)
s7  = color(S7)
s8  = color(S8)
s9  = color(S9)
s10 = color(S10)
s11 = color(S11)
s12 = color(S12)
s13 = color(S13)
s14 = color(S14)
s15 = color(S15)
s16 = color(S16)
s17 = color(S17)

# bold fg
b_s0  = color(S0,  bold=True)
b_s1  = color(S1,  bold=True)
b_s2  = color(S2,  bold=True)
b_s3  = color(S3,  bold=True)
b_s4  = color(S4,  bold=True)
b_s5  = color(S5,  bold=True)
b_s6  = color(S6,  bold=True)
b_s7  = color(S7,  bold=True)
b_s8  = color(S8,  bold=True)
b_s9  = color(S9,  bold=True)
b_s10 = color(S10, bold=True)
b_s11 = color(S11, bold=True)
b_s12 = color(S12, bold=True)
b_s13 = color(S13, bold=True)
b_s14 = color(S14, bold=True)
b_s15 = color(S15, bold=True)
b_s16 = color(S16, bold=True)
b_s17 = color(S17, bold=True)

# fg on bg
bg_s0  = color(BLACK, S0)
bg_s1  = color(BLACK, S1)
bg_s2  = color(BLACK, S2)
bg_s3  = color(BLACK, S3)
bg_s4  = color(BLACK, S4)
bg_s5  = color(BLACK, S5)
bg_s6  = color(BLACK, S6)
bg_s7  = color(BLACK, S7)
bg_s8  = color(BLACK, S8)
bg_s9  = color(BLACK, S9)
bg_s10 = color(BLACK, S10)
bg_s11 = color(BLACK, S11)
bg_s12 = color(BLACK, S12)
bg_s13 = color(BLACK, S13)
bg_s14 = color(BLACK, S14)
bg_s15 = color(BLACK, S15)
bg_s16 = color(BLACK, S16)
bg_s17 = color(BLACK, S17)

# bold fg on bg
bgb_s0  = color(BLACK, S0, bold=True)
bgb_s1  = color(BLACK, S1, bold=True)
bgb_s2  = color(BLACK, S2, bold=True)
bgb_s3  = color(BLACK, S3, bold=True)
bgb_s4  = color(BLACK, S4, bold=True)
bgb_s5  = color(BLACK, S5, bold=True)
bgb_s6  = color(BLACK, S6, bold=True)
bgb_s7  = color(BLACK, S7, bold=True)
bgb_s8  = color(BLACK, S8, bold=True)
bgb_s9  = color(BLACK, S9, bold=True)
bgb_s10 = color(BLACK, S10, bold=True)
bgb_s11 = color(BLACK, S11, bold=True)
bgb_s12 = color(BLACK, S12, bold=True)
bgb_s13 = color(BLACK, S13, bold=True)
bgb_s14 = color(BLACK, S14, bold=True)
bgb_s15 = color(BLACK, S15, bold=True)
bgb_s16 = color(BLACK, S16, bold=True)
bgb_s17 = color(BLACK, S17, bold=True)

# --- ASCII ART ---
import numpy as np
if not hasattr(State, "bands"):
    State.bands = None

import numpy as np

# ---------------- CONFIGURATION ----------------
CFG = {
    "smoothing": 0.8,           # CAVA default smoothing
    "noise_reduction": 0.77,    # noise floor
    "sensitivity": 1.0,         # overall gain
    "min_freq": 20,             # lowest frequency
    "max_freq": 20000,          # highest frequency
    "gravity": 2.0,             # how fast bars fall
    "integral": 0.7,            # smoothing integral
}
# -----------------------------------------------

def wave_fmt(window):
    """
    True CAVA-style audio visualization matching the reference image.
    """
    if not State.is_up:
        return [(s0, "nothing...")]

    return [(s0, f"{State.artist} {State.title} {State.album} {State.frame_count}\n1 = {State.Lpeak}   {State.Lvol}\n2 = {State.Rpeak}   {State.Rvol}")]
    width = window.render_info.window_width
    height = window.render_info.window_height
    
    # CAVA uses the full width as individual bars
    n_bars = width

    # --- FFT Processing (CAVA style) ---
    # Apply Hann window
    windowed_audio = audio * np.hanning(len(audio))
    
    # FFT
    fft_data = np.fft.rfft(windowed_audio)
    magnitude = np.abs(fft_data)
    freqs = np.fft.rfftfreq(len(audio), 1.0 / State.sample_rate)
    
    # --- CAVA Frequency Bin Mapping ---
    # CAVA uses logarithmic frequency distribution
    # Create frequency boundaries for each bar
    freq_boundaries = np.logspace(
        np.log10(CFG["min_freq"]), 
        np.log10(CFG["max_freq"]), 
        n_bars + 1
    )
    
    bar_values = np.zeros(n_bars)
    
    # Map frequency bins to bars (CAVA method)
    for i in range(n_bars):
        freq_low = freq_boundaries[i]
        freq_high = freq_boundaries[i + 1]
        
        # Find FFT bins in this frequency range
        mask = (freqs >= freq_low) & (freqs < freq_high)
        
        if np.any(mask):
            # CAVA takes the sum of magnitudes in the frequency range
            bar_values[i] = np.sum(magnitude[mask])
        else:
            bar_values[i] = 0
    
    # --- CAVA Processing Chain ---
    # Convert to dB-like scale
    bar_values = np.log10(bar_values + 1e-10)
    
    # Normalize
    if bar_values.max() > bar_values.min():
        bar_values = (bar_values - bar_values.min()) / (bar_values.max() - bar_values.min())
    
    # Apply sensitivity
    bar_values *= CFG["sensitivity"]
    
    # Initialize smoothing state
    if not hasattr(State, 'prev_bars') or len(State.prev_bars) != n_bars:
        State.prev_bars = np.zeros(n_bars)
        State.fall_bars = np.zeros(n_bars)
    
    # --- CAVA Smoothing Algorithm ---
    for i in range(n_bars):
        current = bar_values[i]
        
        # Noise reduction
        if current < CFG["noise_reduction"]:
            current = 0
        
        # CAVA smoothing: quick attack, slow release
        if current > State.prev_bars[i]:
            # Rising - quick response
            State.prev_bars[i] = CFG["integral"] * State.prev_bars[i] + (1 - CFG["integral"]) * current
        else:
            # Falling - apply gravity
            State.fall_bars[i] += CFG["gravity"] / height
            State.prev_bars[i] = max(current, State.prev_bars[i] - State.fall_bars[i])
            if State.prev_bars[i] <= current:
                State.fall_bars[i] = 0
    
    # Scale to terminal height
    scaled_bars = (State.prev_bars * height).astype(int)
    scaled_bars = np.clip(scaled_bars, 0, height)
    
    # --- Render exactly like CAVA ---
    lines = []
    for row in range(height):
        line = ""
        for col in range(width):
            if col < len(scaled_bars):
                bar_height = scaled_bars[col]
                # Draw from bottom up (CAVA style)
                if (height - row) <= bar_height:
                    line += "█"
                else:
                    line += " "
            else:
                line += " "
        lines.append(line)
    
    


