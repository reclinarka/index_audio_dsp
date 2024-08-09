import numpy as np
from scipy.optimize import curve_fit
from scipy.io import wavfile
from scipy.fftpack import rfft, ifft, fftfreq
import matplotlib.pyplot as plt
import sys
import soundfile as sf

# inspired by https://www.researchgate.net/publication/236843786_A_psychoacoustic_bass_enhancement_system_with_improved_transient_and_steady-state_performance
# --L--------------------------------------------------------------------------> + ----> Left Channel
#   |                                   (Steady-State Component)                 A
#   |         |>---------------------------------> * ----->  PV  --> IFFT -> G   |
#   |         |                                    A                         |   |
#   V         |                                    |                         V   |
#   + -> FFT -+-> Median Filter -> Transient/Steady-State separation masks   + -<|
#   A         |                                    |                         A   |
#   |         |                                    V                         |   |
#   |         |>---------------------------------> * -----> IFFT --> NLD  -> G   |
#   |                                     (Transient Component)                  V
# --R -------------------------------------------------------------------------> + ----> Right Channel

fft_sample_length = 16_000
fft_sliding_interval = 490  # 44100 / 90


def split_channels(audio, sample_rate):
    l_channel = np.zeros_like(audio)
    r_channel = np.zeros_like(audio)
    return l_channel, r_channel


def median_filter(audio, sampling_rate):
    return np.zeros_like(audio)


def transient_steady_separation(audio, median_filtered, sampling_rate):
    transient_component = np.zeros_like(audio)
    steady_state_component = np.zeros_like(audio)
    return transient_component, steady_state_component


def phase_vocoder(audio, sampling_rate):
    return np.zeros_like(audio)


def non_linear_device(audio, sampling_rate):
    return np.zeros_like(audio)


def G(audio, sampling_rate):  # ????
    return np.zeros_like(audio)


def combine_channels(channel_1, channel_2, sampling_rate):
    return np.zeros_like(channel_1)


def bass_enhance(audio, sampling_rate):
    l_channel, r_channel = split_channels(audio, sampling_rate)
    mono = l_channel + r_channel
    fft_mono = rfft(mono)

    median_filtered = median_filter(fft_mono, sampling_rate)
    transient_component, steady_state_component = transient_steady_separation(fft_mono, median_filtered, sampling_rate)

    # Phase Vocoder Part
    pv_component = phase_vocoder(steady_state_component * fft_mono, sampling_rate)
    pv_component = ifft(pv_component)
    pv_component = G(pv_component, sampling_rate)

    # Non-Linear-Device Part
    nld_component = ifft(transient_component * fft_mono)
    nld_component = non_linear_device(nld_component, sampling_rate)
    nld_component = G(nld_component, sampling_rate)

    # Combine PV and NLD parts
    bass_component = pv_component + nld_component

    l_channel = bass_component + l_channel
    r_channel = bass_component + r_channel

    enhanced_audio = combine_channels(l_channel, r_channel, sampling_rate)
    return enhanced_audio


def main():
    filename = "Teminite & Boom Kitty - The Master [Beat Saber OST 7] [ ezmp3.cc ].mp3"
    audio, sample_rate = sf.read(filename)
    print(f"loaded {filename} with sr of {sample_rate}/s")
    enhanced = bass_enhance(audio, sample_rate)
    return enhanced


if __name__ == '__main__':
    main()
