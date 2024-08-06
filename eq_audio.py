import numpy as np
from scipy.signal import butter, lfilter
import librosa
import soundfile as sf

def read_eq_table(filename):
    data = np.loadtxt(filename, skiprows=1)  # Skip header row
    frequencies = data[:, 0]
    spls = data[:, 1]
    phases = data[:, 2]
    return frequencies, spls, phases


def eq_filter(signal, frequencies, spls, phases, fs):
    # Convert SPL to linear gain
    gains = 10 ** (spls / 20)

    filtered_signal = np.zeros_like(signal)

    for freq, gain, phase in zip(frequencies, gains, phases):
        # Design the bandpass filter
        nyq = 0.5 * fs
        low = freq - (freq / 10.0)
        high = freq + (freq / 10.0)
        low = low / nyq
        high = high / nyq
        b, a = butter(1, [low, high], btype='band')

        # Apply the filter
        filtered_band = lfilter(b, a, signal)

        # Apply gain and phase shift
        filtered_band *= gain
        phase_shift = np.exp(1j * np.deg2rad(phase))
        filtered_band = np.real(filtered_band * phase_shift)

        # Add filtered band to final signal
        filtered_signal += filtered_band

    return filtered_signal



def main():
    filename = '0_mes_eq_vals.txt'
    frequencies, spls, phases = read_eq_table(filename)
    print(frequencies, spls, phases)

    input_file = '256kMeasSweep_0_to_20000_0_dBFS_48k_Float_ref.wav'
    output_file = 'output_audio.wav'
    y, sr = librosa.load(input_file, sr=None)  # Load with original sampling rate

    # Apply EQ filter
    y_eq = eq_filter(y, frequencies, spls, phases, sr)

    # Save the processed audio
    sf.write(output_file, y_eq, sr)

if __name__ == '__main__':
    main()
