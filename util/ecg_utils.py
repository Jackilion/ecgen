import numpy as np
from scipy.signal import firwin, filtfilt, hilbert
from scipy.stats import kurtosis

def detect_ecg_qrs(ecg_vector, samplerate):
    if not isinstance(ecg_vector, np.ndarray) or not np.issubdtype(ecg_vector.dtype, np.number):
        raise ValueError('ecg_vector is not a numeric array')
    
    if not np.isscalar(samplerate) or not np.issubdtype(type(samplerate), np.integer):
        raise ValueError('samplerate must be an integer')
    
    # Determine the best window length. (Should include multiple heart beats)
    nyquistrate = samplerate / 2
    #nyquistrate = 2 / samplerate
    if samplerate < 500:
        window_length = 1024
    elif samplerate < 1000:
        window_length = 2048
    elif samplerate < 2000:
        window_length = 4096
    else:
        window_length = 8192

    # Filter as specified by Benitez et al.
    filter_coeff = firwin(21, [8 / nyquistrate, 20 / nyquistrate], pass_zero='bandstop', window=('kaiser', 21))
    filtered_ecg_gradient = np.gradient(filtfilt(filter_coeff, [1.0], ecg_vector))


    # Iterate through the ecg window by window
    rr_list = []

    ecg_length = len(ecg_vector)
    window_start = 0
    window_end = min(ecg_length, window_start + window_length)
    last_start = -1
    last_max = 0
    last_removed_last_peak = []
    while (window_start < ecg_length) and (window_start > last_start) and (window_end > window_start):
        last_start = window_start
        ecg_segment = ecg_vector[window_start:window_end]
        feg_segment = filtered_ecg_gradient[window_start:window_end]

        transformed_segment = np.imag(hilbert(feg_segment))
        max_transformed = np.max(transformed_segment)
        rms_transformed = np.sqrt(np.mean(np.square(transformed_segment)))

        # There seems to be no signal here...just ignore and continue
        if rms_transformed == 0 or np.isnan(rms_transformed) or np.isnan(max_transformed):
            window_start = window_start + window_length
            window_end = min(ecg_length, window_start + window_length)
            #print("[ECG FILT] Encountered a seemingly empty window. Skipping.")
            continue

        thresholds = []
        if rms_transformed >= 0.18 * max_transformed:
            thresholds.append(0.39 * max_transformed)
        if window_start > 0 and max_transformed > 2 * last_max:
            thresholds.append(0.39 * last_max)
        thresholds.append(1.6 * rms_transformed)
        segment_threshold = max(thresholds)
        last_max = max_transformed

        peak_selection = transformed_segment >= segment_threshold
        peak_selection[0] = False
        peak_selection[-1] = False
        peak_intervals = np.diff(np.concatenate(([0], peak_selection.astype(int), [0])))

        start_interval = np.where(peak_intervals == 1)[0] + 1
        end_interval = np.where(peak_intervals == -1)[0]

        peak_list = []
        n_last_interval = min(len(start_interval), len(end_interval))
        for i in range(n_last_interval):
            if end_interval[i] - start_interval[i] < 3:
                start_interval[i] = start_interval[i] - 1
                end_interval[i] = end_interval[i] + 1
            locs = np.argmax(transformed_segment[start_interval[i]:end_interval[i]])
            peak_list.append(locs + start_interval[i] - 1)

        peak_times = np.array(peak_list) / samplerate
        peak_diff = np.diff(peak_times)
        min_peak_dist = 0.360
        limit_min_peak_dist = 0.200
        if np.mean(peak_diff) - np.std(peak_diff) < min_peak_dist:
            min_peak_dist = max(np.mean(peak_diff) - np.std(peak_diff), limit_min_peak_dist)
        while np.any(peak_diff < 0.360):
            temp = np.where(peak_diff < 0.360)[0]
            removed_peaks = []
            for i in temp:
                peak1 = peak_list[i]
                peak2 = peak_list[i + 1]

                if transformed_segment[peak1] <= transformed_segment[peak2]:
                    removed_peaks.append(i)
                elif transformed_segment[peak2] <= transformed_segment[peak1]:
                    removed_peaks.append(i + 1)

            if not removed_peaks:
                break

            peak_list = np.delete(peak_list, removed_peaks)
            peak_times = peak_list / samplerate
            peak_diff = np.diff(peak_times)

        if len(peak_list) < 2:
            window_start = window_start + int(0.1 * samplerate)
            window_end = min(ecg_length, window_start + window_length)
            continue

        for i in range(len(peak_list)):
            current_peak_start = max(1, peak_list[i] - 10)
            current_peak_end = min(len(ecg_segment), peak_list[i] + 10)

            locs = np.argmax(ecg_segment[current_peak_start:current_peak_end])
            poss_peak_pos = locs + 1

            if poss_peak_pos == 1 or poss_peak_pos == (current_peak_end - current_peak_start + 1):
                continue

            if i < len(peak_list) - 1:
                rr_list.append(int(locs + current_peak_start + window_start - 1))
            else:
                last_removed_last_peak.append(int(locs + current_peak_start + window_start - 1))

        window_start = window_start + peak_list[-1] - int(0.2 * samplerate)
        window_end = min(ecg_length, window_start + window_length)

    if last_removed_last_peak and rr_list[-1] < last_removed_last_peak[0]:
        rr_list.append(last_removed_last_peak[0])

    return np.array(rr_list)
