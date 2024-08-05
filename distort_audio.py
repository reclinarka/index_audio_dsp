import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import sys

max_gain = 0.05

# Function to apply a bandpass filter
def bandpass_filter(audio, lowcut, highcut, sample_rate, order=2):
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, audio, padlen=25)
    return y

# Function to add harmonic distortion
def add_harmonic_distortion(audio, gains, harmonics):
    distorted_audio = np.zeros_like(audio)
    for gain, harmonic in zip(gains, harmonics):
        distorted_audio += gain * np.sin(harmonic * np.arcsin(np.clip(audio,-1,1)))
    return distorted_audio

# Function to interpolate gains
def interpolate_gains(volume, gains_dict):
    volumes = np.array(sorted(gains_dict.keys()))
    gain_values = np.array([gains_dict[v][0] for v in volumes])
    thd_values = np.array([gains_dict[v][1] for v in volumes])
    
    gain_interpolators = [interp1d(volumes, gain_values[:, i], kind='linear', fill_value='extrapolate') for i in range(gain_values.shape[1])]
    thd_interpolator = interp1d(volumes, thd_values, kind='linear', fill_value='extrapolate')
    
    interpolated_thd = thd_interpolator(volume)
    interpolated_gains = [interpolator(volume) * interpolated_thd * max_gain for interpolator in gain_interpolators]
    
    return interpolated_gains, interpolated_thd

def main():
    args = sys.argv
    filename = args[1]
    # Load the audio file
    audio, sample_rate = sf.read(filename)

    # Ensure the audio is a 1D array (in case of stereo, take one channel)
    if audio.ndim > 1:
        audio = audio[:, 0]

    # Harmonic distortion gains and THD values
    gains_dict = {
        0:      ([0.0128, 0.261, 0.0086, 0.0876, 0.00452, 0.02, 0.0012, 0.0012], 0.273),
        -1.25:  ([0.0101, 0.257, 0.00674, 0.0832, 0.00298, 0.0166, 0.00048, 0.0293], 0.269),
        -2.5:   ([0.0134, 0.251, 0.00866, 0.0755, 0.00365, 0.0115, 0.00051, 0.00488], 0.263),
        -3.75:  ([0.0145, 0.24, 0.00883, 0.0636, 0.00298, 0.00314, 0.00043, 0.00864], 0.249),
        -5:     ([0.016, 0.217, 0.00894, 0.0397, 0.00271, 0.0093, 0.001, 0.0102], 0.222),
        -6.25:  ([0.00513, 0.192, 0.00287, 0.0179, 0.00089, 0.0164, 0.00023, 0.00767], 0.194),
        -7.5:   ([0.00191, 0.157, 0.00218, 0.00592, 0.00161, 0.0177, 0.00041, 0.00143], 0.158),
        -8.75:  ([0.00535, 0.111, 0.00237, 0.0245, 0.00034, 0.00904, 0.00037, 0.00481], 0.114),
        -10:    ([0.00472, 0.0574, 0.00155, 0.0276, 0.00045, 0.00427, 0.00028, 0.00244], 0.064),
        -11.25: ([0.00584, 0.00745, 0.0023, 0.00653, 0.00086, 0.00401, 0.00031, 0.00237], 0.0126),
        -12.25: ([0.00662, 0.00231, 0.00038, 0.00032, 0.00041, 0.0003, 0.00041, 0.00039], 0.00707),
    }


    # Parameters
    lowcut = 40.0
    highcut = 130.0
    harmonics = [2, 3, 4, 5, 6, 7, 8, 9]



    # Determine the input volume (e.g., RMS value in dB)
    rms_value = np.sqrt(np.mean(audio**2))
    input_volume = 20 * np.log10(rms_value)

    # Interpolate appropriate gains based on input volume
    interpolated_gains, interpolated_thd = interpolate_gains(input_volume, gains_dict)

    # Ensure the audio is long enough for the filter (padlen=25 is the padding length)
    if len(audio) <= 25:
        raise ValueError("The length of the input audio must be greater than 25 samples.")

    # Isolate the fundamental frequencies in the range of 50 to 100 Hz
    filtered_audio = bandpass_filter(audio, lowcut, highcut, sample_rate)

    # Apply harmonic distortion only to the filtered audio
    distorted_filtered_audio = add_harmonic_distortion(filtered_audio, interpolated_gains, harmonics)

    # Combine the distorted signal with the original signal
    combined_audio = audio + distorted_filtered_audio

    # Normalize to avoid clipping
    combined_audio /= np.max(np.abs(combined_audio))

    # Save the processed audio
    sf.write( f'distorted_{lowcut}-{highcut}_{max_gain}_' + filename, combined_audio, sample_rate)


if __name__ == "__main__":
    main()
    # & 'C:\Users\recli\anaconda3\shell\condabin\conda-hook.ps1' ; conda activate 'C:\Users\recli\anaconda3'