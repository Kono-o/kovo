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
        fg      = color(hexcode),
        ital    = color(hexcode, italic=True),
        b_fg    = color(hexcode, bold=True),
        bg      = color(BLACK, hexcode),
        b_bg    = color(BLACK, hexcode, bold=True),
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

import numpy as np
from .state import State

# --- SETTINGS ---
BAR_WIDTH = 2
BAR_GAP = 1
GAIN = 0.5
PREEMPHASIS_COEFFICIENT = 0.97  # A common value for pre-emphasis filters
SMOOTHING_FACTOR = 0.4
BASS_SMOOTHING = 0.2    # Slower response for bass
TREBLE_SMOOTHING = 0.6  # Faster response for treble
MID_RANGE_SMOOTHING = 0.8 # Fastest response for the middle
RISE_SPEED = 0.4
FALL_SPEED = 0.2
MIN_FREQ = 50
MAX_FREQ = 10000
FFT_SIZE = 4096
LINEAR_BARS = 3       # Number of bars to handle linearly
LINEAR_FREQ_CUTOFF = 200 # Frequency (Hz) at which to switch from linear to log scale
BASE_RISE_SPEED = 0.4
BASE_FALL_SPEED = 0.2
TREBLE_FALL_FACTOR = 0.8 # Higher value makes treble fall faster
TREBLE_RISE_FACTOR = 1.5 # Higher value makes treble rise faster
BAR_CHARS = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']


prev_bars = []

# --- LOGARITHMIC BIN MAPPING (safe) ---
def get_cava_bins(sample_rate, fft_size, num_bars):
    freqs = np.fft.rfftfreq(fft_size, 1 / sample_rate)

    # 1. Linear part for the lowest frequencies
    linear_bins = np.linspace(MIN_FREQ, LINEAR_FREQ_CUTOFF, LINEAR_BARS + 1)

    # 2. Logarithmic part for the rest
    log_bins = np.logspace(np.log10(LINEAR_FREQ_CUTOFF), np.log10(MAX_FREQ), num_bars - LINEAR_BARS + 1)

    # Combine and remove the overlapping point
    all_bins = np.concatenate([linear_bins[:-1], log_bins])

    bin_indices = np.searchsorted(freqs, all_bins)
    bin_indices = np.clip(bin_indices, 1, len(freqs) - 1)
    
    return bin_indices

# --- FFT PROCESSING (revised) ---
def process_fft(audio_data, sample_rate, num_bars):
    if audio_data is None or len(audio_data) < 2:
        return np.zeros(num_bars)

    # Apply a pre-emphasis filter to the audio data
    pre_emphasized = np.zeros_like(audio_data)
    pre_emphasized[1:] = audio_data[1:] - PREEMPHASIS_COEFFICIENT * audio_data[:-1]
    
    # Pad or slice to FFT_SIZE
    if len(pre_emphasized) < FFT_SIZE:
        padded = np.zeros(FFT_SIZE)
        padded[:len(pre_emphasized)] = pre_emphasized
    else:
        padded = pre_emphasized[:FFT_SIZE]

    windowed = padded * np.hanning(FFT_SIZE)
    fft = np.fft.rfft(windowed)
    magnitude = np.abs(fft)

    # Use a simple logarithmic binning for a smooth distribution
    freqs = np.fft.rfftfreq(FFT_SIZE, 1/sample_rate)
    log_bins = np.logspace(np.log10(MIN_FREQ), np.log10(MAX_FREQ), num_bars + 1)
    bin_indices = np.searchsorted(freqs, log_bins)
    bin_indices = np.clip(bin_indices, 1, len(freqs)-1)

    bars = np.zeros(num_bars)
    for i in range(num_bars):
        start, end = bin_indices[i], bin_indices[i+1]
        if end > start:
            bars[i] = np.mean(magnitude[start:end])
        else:
            bars[i] = magnitude[start]

    max_val = np.max(bars)
    if max_val > 1e-6:
        bars /= max_val

    return bars

# --- CAVA SMOOTHING ---
def smooth_bars(current, previous):
    if len(previous) != len(current):
        return current.copy()

    smoothed = np.zeros_like(current)
    num_bars = len(current)
    center_bar = num_bars // 2

    for i in range(num_bars):
        # A simple model for frequency-dependent smoothing
        if i < num_bars * 0.2: # First 20% of bars (bass)
            smooth_factor_i = SMOOTHING_FACTOR * BASS_SMOOTHING
        elif i > num_bars * 0.8: # Last 20% of bars (treble)
            smooth_factor_i = SMOOTHING_FACTOR * TREBLE_SMOOTHING
        else: # Middle 60% of bars
            smooth_factor_i = SMOOTHING_FACTOR * MID_RANGE_SMOOTHING
            
        if current[i] > previous[i]:
            # Apply a higher gain/rise speed for the middle bars
            rise_gain = 1.0 + (1.5 if i > num_bars * 0.2 and i < num_bars * 0.8 else 0)
            smoothed[i] = previous[i] + (current[i] - previous[i]) * rise_gain * (1 - smooth_factor_i)
        else:
            # Slower fall speed
            smoothed[i] = previous[i] * (1 - SMOOTHING_FACTOR) + current[i] * SMOOTHING_FACTOR
        
        # Final smoothing based on overall factor
        smoothed[i] = previous[i] * (1 - SMOOTHING_FACTOR) + smoothed[i] * SMOOTHING_FACTOR

    return smoothed

# --- DISPLAY HEIGHT ---
def bars_to_display(bars, height):
    return np.clip((bars * GAIN * (height-1)).astype(int), 0, height-1)

# --- RENDER ---
def render_bars(bar_heights, width, height):
    display_lines = []
    num_bars = len(bar_heights)

    for row in range(height):
        line = ""
        for i in range(num_bars):
            bar_height = bar_heights[i]
            display_row = height - 1 - row
            char = BAR_CHARS[-1] if display_row < bar_height else BAR_CHARS[0]
            line += char * BAR_WIDTH
            if i < num_bars - 1:
                line += " " * BAR_GAP
        display_lines.append((s0, line.ljust(width) + "\n"))
    return display_lines

# --- MAIN FORMATTER ---
def wave_fmt(window):
    global prev_bars

    if not State.is_up or State.Ln is None:
        return [(s0, "waiting for audio...")]

    width = window.render_info.window_width
    height = window.render_info.window_height
    if width <= 0 or height <= 0:
        return [(s0, "invalid dimensions")]

    num_bars = max(1, width // (BAR_WIDTH + BAR_GAP))
    audio_data = (State.Ln + State.Rn) * 0.5 if State.Rn is not None else State.Ln
    #print("audio max:", np.max(np.abs(audio_data)), "len:", len(audio_data))
    current = process_fft(audio_data, State.sample_rate, num_bars)
    if len(prev_bars) != num_bars:
        prev_bars = np.zeros(num_bars)

    smoothed = smooth_bars(current, prev_bars)
    prev_bars = smoothed.copy()

    bar_heights = bars_to_display(smoothed, height)
    return render_bars(bar_heights, width, height)