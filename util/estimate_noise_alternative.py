import numpy as np

def estimate_ecg_noise(ecg_signal, r_peaks, fs, num_components=3):
    """
    Estimate the noise in an ECG signal using SVD decomposition.

    Parameters:
    - ecg_signal: numpy array of ECG signal samples.
    - r_peaks: numpy array of indices of R-peak locations in the ECG signal.
    - fs: Sampling frequency in Hz.
    - num_components: Number of singular values/components to keep in reconstruction.

    Returns:
    - noise_signal: numpy array of the estimated noise signal.
    - snr_db: Signal-to-Noise Ratio in decibels.
    """

    # Define window in samples: -50 ms to +100 ms around each R-peak
    pre_r_peak = int(0.5 * fs)   # 50 ms before R-peak
    post_r_peak = int(0.5 * fs)  # 100 ms after R-peak
    window_length = pre_r_peak + post_r_peak  # Total window length

    #normalise signal
    #ecg_signal = (ecg_signal - ecg_signal.min()) / ((ecg_signal.max() - ecg_signal.min()) + 1e-6)

    # Collect segments around R-peaks
    segments = []
    valid_r_peaks = []
    for r_peak in r_peaks:
        start_idx = r_peak - pre_r_peak
        end_idx = r_peak + post_r_peak
        # Ensure indices are within the signal boundaries
        if start_idx >= 0 and end_idx <= len(ecg_signal):
            segment = ecg_signal[start_idx:end_idx]
            segments.append(segment)
            valid_r_peaks.append(r_peak)
        else:
            # Skip if the window exceeds signal boundaries
            continue

    # Check if any valid segments are collected
    if not segments:
        print("No valid segments found. Please check your R-peak indices and signal length.")
        noise_signal = ecg_signal.copy()
        snr_db = np.nan
        return noise_signal, snr_db

    # Convert list to numpy array and transpose so each column is a segment
    segments = np.array(segments).T  # Shape: (window_length, num_segments)

    # Perform SVD on the segments matrix
    U, S, Vt = np.linalg.svd(segments, full_matrices=False)

    # Reconstruct the segments using the specified number of singular values/components
    U_reduced = U[:, :num_components]
    S_reduced = np.diag(S[:num_components])
    Vt_reduced = Vt[:num_components, :]
    reconstructed_segments = U_reduced @ S_reduced @ Vt_reduced

    # Map the reconstructed segments back to their positions in the signal
    reconstructed_signal = np.zeros_like(ecg_signal, dtype=float)
    counts = np.zeros_like(ecg_signal, dtype=int)

    for i, r_peak in enumerate(valid_r_peaks):
        start_idx = r_peak - pre_r_peak
        end_idx = r_peak + post_r_peak
        reconstructed_signal[start_idx:end_idx] += reconstructed_segments[:, i]

        counts[start_idx:end_idx] += 1

    # Avoid division by zero
    counts_nonzero = np.where(counts == 0, 1, counts)
    if counts_nonzero.min() < 0:
        print("OOF!")

    # Average overlapping regions
    reconstructed_signal /= counts_nonzero

    # Estimate the noise signal
    noise_signal = ecg_signal - reconstructed_signal
    #noise_signal = reconstructed_signal
    # Identify noise-only regions (outside R-peak windows)
    noise_only_indices = counts == 0

    # Compute signal power in regions with R-peaks
    signal_indices = counts > 0
    start_idx = int(fs / 2)
    end_idx = len(ecg_signal) - int(fs / 2)
    signal_power = np.mean(ecg_signal[start_idx : end_idx] ** 2)
    noise_power = np.mean(noise_signal[start_idx: end_idx] ** 2)
    # Calculate SNR in decibels
    snr_db = 10 * np.log10(signal_power / noise_power) if noise_power != 0 else np.inf

    return noise_signal, snr_db
