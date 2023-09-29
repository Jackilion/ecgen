import numpy as onp
from tqdm import tqdm
from hrv import calculate_time_domain_parameters
from util.adaptive_filter_new import adaptive_hrv_filter

from util.ecg_utils import detect_ecg_qrs
from util.hrv_utils import calculate_frequency_domain_parameters
#! Load the dataset


def filter_batch(batch, batch_nr, min_hr, max_hr):
    batch_params_td = []
    batch_params_freq = []
    batch_noNNTime =  []
    
    out_batch = []
    pbar = tqdm(range(batch.shape[0]), desc=f"Batch number {batch_nr}")
    for i in pbar:
        ecg = batch[i]
        r_peaks = detect_ecg_qrs(ecg, samplerate=1024)
        r_peaks_time = r_peaks / 1024 * 1000
        r_peak_diff = onp.abs(onp.diff(r_peaks_time, 1))

        filtered, noNN_time = adaptive_hrv_filter(r_peak_diff)
        filtered = onp.array(filtered)

        
        batch_noNNTime.append(noNN_time)


        td_params = calculate_time_domain_parameters(hrv_series=filtered)      
        if td_params["hr"] > min_hr and td_params["hr"] < max_hr:
            out_batch.append(ecg)
    
    return out_batch


data = onp.load("data/ecgs_1024.npy")


P, M, S = data.shape #Patients, minutes, samples

data_2d = data.reshape(P*M, S)
max_val = onp.max(data_2d)
min_val = onp.min(data_2d)
range_vals = max_val - min_val
data_2d = (data_2d + onp.abs(min_val)) / (max_val + onp.abs(min_val))


print(data_2d.shape)
data_cut = data_2d.reshape(54, 64, 327_680)


output_dataset = []
for i in range(data_cut.shape[0]):
    raw_batch = data_cut[i]
    selected_ecgs =  filter_batch(raw_batch, i, 65, 75)
    
    if (len(selected_ecgs) < 64):
        q, r = divmod(64, len(selected_ecgs))
        repeated = q * selected_ecgs + selected_ecgs[:r]
    else:
        repeated = selected_ecgs
        
    
    output_dataset.append(repeated)
    
onp.save("ecgs_within_65_and_75_hr.npy", onp.array(output_dataset))
print(onp.array(output_dataset).shape)