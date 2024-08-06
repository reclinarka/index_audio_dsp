import numpy as np
from scipy.io import wavfile
from scipy.fftpack import fft, ifft, fftfreq

def parse_table(filename):
    data = {}
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.strip():
                values = line.split()
                if values[0] == "*": continue
                freq = float(values[0])
                data[freq] = {
                    'fundamental': float(values[1]),
                    'thd': float(values[2]),
                    'noise': float(values[3]),
                    'harmonics': [float(v) for v in values[4:]]
                }
    return data


def db_to_amplitude(db):
    """Convert dB value to amplitude."""
    return 10 ** (db / 20)


def add_harmonics(signal, sample_rate, harmonics):
    """Add harmonics to a signal."""
    t = np.linspace(0, len(signal) / sample_rate, num=len(signal))
    harmonic_signal = np.zeros_like(signal)
    for i, level in enumerate(harmonics):
        if level != float('-inf'):  # Skip if the level is negative infinity (no harmonic)
            amplitude = db_to_amplitude(level)
            harmonic_signal += amplitude * np.sin(2 * np.pi * (i + 2) * t)  # (i + 2) to match H2, H3, ...
    return signal + harmonic_signal


def apply_harmonic_distortion(data, sample_rate, table_data):
    """Apply harmonic distortion based on table data."""
    N = len(data)
    t = np.arange(N) / sample_rate

    # FFT of the original signal
    freqs = fftfreq(N, 1 / sample_rate)
    fft_data = fft(data)
    distorted_fft = np.copy(fft_data)

    for freq, values in table_data.items():
        idx = np.argmin(np.abs(freqs - freq))
        if idx >= N // 2:
            continue

        harmonics = values['harmonics']
        fundamental_amp = np.abs(fft_data[idx])

        # Apply harmonics
        for h_idx, h_level in enumerate(harmonics, start=2):
            harmonic_idx = idx * h_idx
            if harmonic_idx < N // 2:
                amplitude = fundamental_amp * db_to_amplitude(h_level)
                phase = np.angle(fft_data[idx])
                distorted_fft[harmonic_idx] += amplitude * np.exp(1j * phase)
                distorted_fft[-harmonic_idx] += amplitude * np.exp(-1j * phase)

    # Apply noise floor if specified
    noise_floor = db_to_amplitude(values['noise'])
    noise = np.random.normal(0, noise_floor, N)
    distorted_signal = ifft(distorted_fft).real + noise

    return distorted_signal


def process_audio(input_filename, output_filename, table_data):
    sample_rate, data = wavfile.read(input_filename)
    if data.ndim > 1:
        data = data.mean(axis=1)  # Convert to mono if necessary
    distorted_data = apply_harmonic_distortion(data, sample_rate, table_data)
    distorted_data = np.int16(distorted_data / np.max(np.abs(distorted_data)) * 32767)
    wavfile.write(output_filename, sample_rate, distorted_data)

def main():
    table_data = parse_table('0_mes_distortion_dBr.txt')
    process_audio('256kMeasSweep_0_to_20000_0_dBFS_48k_Float_ref.wav', 'output.wav', table_data)

if __name__ == '__main__':
    main()
