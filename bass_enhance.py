import soundfile as sf
import numpy as np
import scipy.signal as signal
import scipy.fftpack as fft
from scipy.ndimage import median_filter, shift
import random

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

transient_weight = 0.1
steady_weight = 00.15
original_weight = 1


def design_biquad_filter(filter_type, freq, Q, gain_db, fs):
    """Design a biquad filter."""
    A = 10 ** (gain_db / 40)
    omega = 2 * np.pi * freq / fs
    alpha = np.sin(omega) / (2 * Q)

    if filter_type == "peaking":
        b0 = 1 + alpha * A
        b1 = -2 * np.cos(omega)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(omega)
        a2 = 1 - alpha / A
    elif filter_type == "low_shelf":
        beta = np.sqrt(A) / Q
        b0 = A * ((A + 1) - (A - 1) * np.cos(omega) + 2 * beta * np.sin(omega))
        b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(omega))
        b2 = A * ((A + 1) - (A - 1) * np.cos(omega) - 2 * beta * np.sin(omega))
        a0 = (A + 1) + (A - 1) * np.cos(omega) + 2 * beta * np.sin(omega)
        a1 = -2 * ((A - 1) + (A + 1) * np.cos(omega))
        a2 = (A + 1) + (A - 1) * np.cos(omega) - 2 * beta * np.sin(omega)
    elif filter_type == "high_shelf":
        beta = np.sqrt(A) / Q
        b0 = A * ((A + 1) + (A - 1) * np.cos(omega) + 2 * beta * np.sin(omega))
        b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(omega))
        b2 = A * ((A + 1) + (A - 1) * np.cos(omega) - 2 * beta * np.sin(omega))
        a0 = (A + 1) - (A - 1) * np.cos(omega) + 2 * beta * np.sin(omega)
        a1 = 2 * ((A - 1) - (A + 1) * np.cos(omega))
        a2 = (A + 1) - (A - 1) * np.cos(omega) - 2 * beta * np.sin(omega)
    else:
        raise ValueError("Invalid filter type")

    b = [b0, b1, b2]
    a = [a0, a1, a2]
    b = np.array(b) / a0
    a = np.array(a) / a0
    return b, a


def apply_equalizer(in_signal, fs, eq_params):
    """Apply a series of biquad filters to the signal."""
    for filter_type, freq, Q, gain_db in eq_params:
        b, a = design_biquad_filter(filter_type, freq, Q, gain_db, fs)
        in_signal = signal.lfilter(b, a, in_signal)
    return in_signal


def median_filter_separation(signal_spectrum, kernel_size=3):
    """Separate transient and steady-state components using median filter."""
    transient = median_filter(signal_spectrum, size=(1, kernel_size))
    steady_state = median_filter(signal_spectrum, size=(kernel_size, 1))
    return transient, steady_state


def apply_nld(transient, coefficients=None):
    """Apply Nonlinear Device (NLD) to transient component."""
    # Example NLD function: Polynomial expansion
    print(transient.shape)
    if coefficients is None:
        # Example polynomial coefficients (these should ideally be derived or specified)

        coefficients = [100/i for i in range(1,10)]  # Example coefficients for polynomial expansion
        #coefficients = [random.uniform(0, 1) for i in range(0,100)]  # Example coefficients for polynomial expansion
        #print(coefficients)

    output = np.zeros_like(transient)
    for i, h_i in enumerate(coefficients):
        output += h_i * np.power(transient*0.99, i)

    output = 1./output
    return output / max(np.abs(output))

def apply_improved_pv(steady_state, harmonic_order=2):
    """Apply improved Phase Vocoder (PV) to steady-state component."""
    # Example Phase Vocoder harmonic generation
    freq_domain = fft.fft(steady_state)
    shifted_freq = shift(freq_domain, harmonic_order, cval=0)
    return np.clip(fft.ifft(shifted_freq).real, -1, 1)


def combine_signals(enhanced_transient, enhanced_steady_state, sampling_rate):
    """Combine original signal with enhanced transient and steady-state components."""
    eq_params = [
        # ("peaking", 120, 1.8, 3.8),
        ("peaking", 1_000, 1.499, -2.6),
        ("peaking", 7_120, 0.5, -6.3),
        ("low_shelf", 50, 0.707, -1.9),
        ("high_shelf", 2_200, 0.707, -0.6)
    ]

    min_length = min(len(enhanced_transient), len(enhanced_steady_state))
    bass_effect = enhanced_transient[:min_length] * transient_weight + enhanced_steady_state[
                                                                       :min_length] * steady_weight
    bass_effect = apply_equalizer(bass_effect, sampling_rate, eq_params)
    return bass_effect


def enhance_bass(input_signal, sampling_rate, kernel_size=5, harmonic_order=2):
    """Enhance bass of the input signal using the proposed hybrid method."""
    # Step 1: Perform Short-Time Fourier Transform (STFT)
    print("Step 1: Perform Short-Time Fourier Transform (STFT)")
    f, t, Zxx = signal.stft(input_signal, fs=sampling_rate, nperseg=1024)
    signal_spectrum = np.abs(Zxx)

    # Step 2: Separate signal into transient and steady-state components
    print("Step 2: Separate signal into transient and steady-state components")
    transient, steady_state = median_filter_separation(signal_spectrum, kernel_size)

    # Step 3: Apply NLD to transient component
    print("Step 3: Apply NLD to transient component")
    _, reconstructed_transient = signal.istft(transient, fs=sampling_rate)
    enhanced_transient = apply_nld(reconstructed_transient)

    # Step 4: Apply improved PV to steady-state component
    print("Step 4: Apply improved PV to steady-state component")
    enhanced_steady_state = apply_improved_pv(steady_state, harmonic_order)

    # Step 5: Inverse STFT to reconstruct time-domain signals
    print("Step 5: Inverse STFT to reconstruct time-domain signals")
    _, reconstructed_steady_state = signal.istft(enhanced_steady_state, fs=sampling_rate)

    # Step 6: Combine signals
    print("Step 6: Combine signals")
    enhanced_signal = combine_signals(enhanced_transient, reconstructed_steady_state, sampling_rate)

    return enhanced_signal


def main():
    mes = 0
    if not mes:
        filename = "Teminite & Boom Kitty - The Master [Beat Saber OST 7] [ ezmp3.cc ].mp3"
        out_file = "master_enhanced.wav"
        out_format = "wav"
    else:
        filename = "256kMeasSweep_0_to_20000_0_dBFS_48k_Float_ref.wav"
        out_file = "mes_enhanced.wav"
        out_format = "wav"

    eq_params = [
        ("peaking", 85, 0.6, 12),
        ("peaking", 174.5, 0.7, -8.6),
        ("peaking", 590, 0.5, 1.5),
        ("peaking", 3_020, 2.29, -12),
        ("peaking", 4_270, 1.499, -1.6),
        ("peaking", 5_980, 2.448, -11.5),
        ("peaking", 8_310, 4.031, 4.4),
        ("low_shelf", 2, 1.6, -120),
        ("high_shelf", 20_000, 0.5, -25)
    ]

    pv_ho = 2
    kernel_size = 10

    input_signal, sampling_rate = sf.read(
        filename)  # Example 10-second random signal

    if input_signal.ndim > 1:
        mono_signal = input_signal[:, 0] + input_signal[:, 1]
        bass_signal = enhance_bass(input_signal[:, 0], sampling_rate, kernel_size=kernel_size, harmonic_order=pv_ho)
        min_length = min(len(mono_signal), len(bass_signal))

        print("Step 7: Enhance signal with Bass and apply EQ")
        enhanced_l_signal = input_signal[:, 0][:min_length] * original_weight + bass_signal[:min_length]
        enhanced_l_signal = apply_equalizer(enhanced_l_signal, sampling_rate, eq_params)
        #enhanced_l_signal = apply_equalizer(enhanced_l_signal, sampling_rate, eq_params)

        enhanced_r_signal = input_signal[:, 1][:min_length] * original_weight + bass_signal[:min_length]
        enhanced_r_signal = apply_equalizer(enhanced_r_signal, sampling_rate, eq_params)
        #enhanced_r_signal = apply_equalizer(enhanced_r_signal, sampling_rate, eq_params)

        max_amp = max(max(np.abs(enhanced_l_signal)), max(np.abs(enhanced_r_signal)), max(np.abs(mono_signal)))

        enhanced_signal = np.vstack((enhanced_l_signal, enhanced_r_signal)).T
        enhanced_signal /= max_amp

    else:
        bass_signal = enhance_bass(input_signal, sampling_rate, kernel_size=kernel_size, harmonic_order=pv_ho)
        min_length = min(len(input_signal), len(bass_signal))
        enhanced_signal = input_signal[:min_length] * original_weight + bass_signal[:min_length]
        enhanced_signal = apply_equalizer(enhanced_signal, sampling_rate, eq_params)

        max_amp = max(np.abs(enhanced_signal))
        enhanced_signal /= max_amp

    # Save or play the enhanced signal as needed
    sf.write(out_file, enhanced_signal, sampling_rate, format=out_format)


if __name__ == "__main__":
    main()
