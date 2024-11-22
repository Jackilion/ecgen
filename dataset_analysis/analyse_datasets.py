from dataset_loader import load_ecg_dataset
import numpy as onp
from util.hrv_utils import calculate_time_domain_parameters, calculate_frequency_domain_parameters
from util.adaptive_filter_new import adaptive_hrv_filter
from scipy.signal import welch
from util.ecg_utils import detect_ecg_qrs
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import jax
import os
import warnings
from scipy.stats import mannwhitneyu

def plot_param_hist(data, title, xlabel, ylabel, name):
    plt.figure()
    plt.hist(data, weights=onp.zeros_like(data) + 1. / len(data), bins=50)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(name)


def param_dict_to_json(params_td, params_freq, nonntime):
    json = \
    {
    "rmssd": onp.average([obj.rmssd for obj in params_td]),
    "rmssd_std": onp.std([obj.rmssd for obj in params_td]),
    "sdnn": onp.average([dictionary.sdNN for dictionary in params_td]),
    "sdnn_std": onp.std([dictionary.sdNN for dictionary in params_td]),
    "hr": onp.average([dictionary.hr for dictionary in params_td]),
    "hr_std": onp.std([dictionary.hr for dictionary in params_td]),
    "shannon": onp.average([obj.shannon for obj in params_td]),
    "shannon_std": onp.std([obj.shannon for obj in params_td]),
    
    "P": onp.average([dictionary.P for dictionary in params_freq]),
    "P_std": onp.std([dictionary.P for dictionary in params_freq]),
    "LF_TO_P": onp.average([dictionary.LF_TO_P for dictionary in params_freq]),
    "LF_TO_P_std": onp.std([dictionary.LF_TO_P for dictionary in params_freq]),
    "HF_TO_P": onp.average([dictionary.HF_TO_P for dictionary in params_freq]),
    "HF_TO_P_std": onp.std([dictionary.HF_TO_P for dictionary in params_freq]),
    "VLF_TO_P": onp.average([dictionary.VLF_TO_P for dictionary in params_freq]),
    "VLF_TO_P_std": onp.std([dictionary.VLF_TO_P for dictionary in params_freq]),
    "LF_TO_HF": onp.average([obj.LF_TO_HF for obj in params_freq]),
    "LF_TO_HF_std": onp.std([obj.LF_TO_HF for obj in params_freq]),
        
    "noNNTime": onp.average(nonntime),
    "noNNTime_std": onp.std(nonntime)
        
    }
    return json

def analyse_batch(batch, batch_nr, savepath):
    batch_params_td = []
    batch_params_freq = []
    batch_noNNTime =  []
    
    pbar = tqdm(range(batch.shape[0]), desc=f"Batch number {batch_nr}")
    for i in pbar:
        ecg = batch[i]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            r_peaks = detect_ecg_qrs(ecg, samplerate=1024)
            r_peaks_time = r_peaks / 1024 * 1000
            r_peak_diff = onp.abs(onp.diff(r_peaks_time, 1))

            filtered, noNN_time = adaptive_hrv_filter(r_peak_diff)
            filtered = onp.array(filtered)

        
        batch_noNNTime.append(noNN_time)
    

        td_params = calculate_time_domain_parameters(hrv_series=filtered)      
        freq_params = calculate_frequency_domain_parameters(filtered)
        
        batch_params_td.append(td_params)
        batch_params_freq.append(freq_params)
        
        
        #save 10% of generated ECGs
        
        script_path = os.path.dirname(__file__)
        save_img = os.path.join(script_path, savepath)
            
        if not os.path.isdir(save_img):
            os.makedirs(save_img)
        if onp.random.uniform(0.0, 1.0) < 0.1:
            
            #get the rpeaks that are visible:

            for j in range(len(r_peaks)):
                if r_peaks[j] > (1024 * 60):
                    break
            y_peaks = []
            for pos in r_peaks:
                y_peaks.append(ecg[pos])
            plt.figure()
            plt.plot(ecg[0:1024*60])
            plt.xlabel("Samples")
            plt.ylabel("AU")
            plt.scatter(r_peaks[0:j], y_peaks[0:j], marker="x", color="red")
            plt.title("1 min ECG and detected R-peaks")
            plt.tight_layout()
            plt.savefig( savepath + f"/ecg_{i}_first_minute_annotated.png")
            
            x = onp.cumsum(filtered)
            x_new = onp.arange(start=0, stop=int(x[-1]), step=1000)
            y = filtered
            x = x * 0.01
            x_new = x_new * 0.01
            interpolated = onp.interp(x_new, x, y)    
            freq, pow = welch(interpolated)
            plt.figure()
            plt.plot(freq, pow)
            plt.title("Power spectrum of 320s ECG")
            plt.xlabel("Frequency [Hertz]")
            plt.ylabel("Power [ms²]")
            plt.tight_layout()
            plt.savefig(savepath + f"/ecg_{i}_five_minute_spectrum.png")
            
            
            plt.figure()
            plt.plot(ecg[0:1024*10])
            plt.xlabel("Samples")
            plt.title("10 seconds ECG")
            plt.ylabel("AU")
            plt.tight_layout()
            plt.savefig(savepath + f"/ecg_{i}_first_10_seconds.png")
            plt.close("all")

        
    return batch_params_td, batch_params_freq, batch_noNNTime



#! Generated Dataset

data = onp.load("generated_datasets/mu_zero_sigma_13.npy")
data = data.reshape(54, 64, -1)

all_params_td_fake = []
all_params_freq_fake = []
all_nonntime_fake = []
for i in range(data.shape[0]):
    batch = data[i]

    batch_params_td, batch_params_freq , batch_noNNTime = analyse_batch(batch, i, f"outputs/analysis/fake/img/batch_{i}")
    
    all_params_freq_fake.extend(batch_params_freq)
    all_params_td_fake.extend(batch_params_td)
    all_nonntime_fake.extend(batch_noNNTime)



all_hr_fake = [obj.hr for obj in all_params_td_fake]
all_rmssd_fake = [obj.rmssd for obj in all_params_td_fake]
all_lf_to_hf_fake = [obj.LF_TO_HF for obj in all_params_freq_fake]
all_nonn_time_fake = all_nonntime_fake
all_lf_to_p_fake = [obj.LF_TO_P for obj in all_params_freq_fake]
all_hf_to_p_fake = [obj.HF_TO_P for obj in all_params_freq_fake]
all_vlf_to_p_fake = [obj.VLF_TO_P for obj in all_params_freq_fake]
all_shannon_fake = [obj.shannon for obj in all_params_td_fake]

json_output = param_dict_to_json(all_params_td_fake, all_params_freq_fake, all_nonn_time_fake)
out_file = open(f"outputs/analysis/fake/stats.json", "x")
json.dump(json_output, out_file, indent=6)
out_file.close()
    
plot_param_hist(all_hr_fake, "Mean heart rate of generated 320s ECGs", "Heart rate [BPM]", "Relative frequency", "fake_hr.png")
plot_param_hist(all_rmssd_fake, "Rmssd of generated 320s ECGs", "rmssd [ms]", "Relative frequency", "fake_rmssd.png")
plot_param_hist(all_lf_to_hf_fake, "LF to HF ratio of generated 320s ECGs", "LF/HF" , "Relative frequency", "fake_lf_to_hf.png")
plot_param_hist(all_nonn_time_fake, "Filter activity of generated 320s ECGs", "NoNN time [ms]", "Relative frequency", "fake_nonntime.png")
plot_param_hist(all_lf_to_p_fake,  "LF to P ratio of generated 320s ECGs", "LF / P", "Relative frequency", "fake_lf_to_p.png")
plot_param_hist(all_hf_to_p_fake, "HF to P ratio of generated 320s ECGs", "HF / P", "Relative frequency", "fake_Hf_to_p.png")
plot_param_hist(all_vlf_to_p_fake, "VLF to P ratio of generated 320s ECGs" , "VLF / P", "Relative frequency", "fake_vlf_to_p.png")
plot_param_hist(all_shannon_fake, "Shannon Entropy of generated 320s ECGs", "Shannon Entropy", "Relative Frequency", "fake_shannon.png")


#!##Dataset###
del data
#data = onp.load(".npy")
rng = jax.random.PRNGKey(0)
data, labels = load_ecg_dataset(rng, 1024 * 5 * 64, batch_size=64, normalise=True)
# data = onp.load("generated_datasets/mu_zero_sigma_01.npy")
# data = data.reshape(54, 64, -1)

data = onp.array(data) # move it to the cpu
all_params_td_real = []
all_params_freq_real = []
all_nonn_time_real = []
for i in range(data.shape[0]):
    batch = data[i]

    batch_params_td, batch_params_freq , batch_noNNTime = analyse_batch(batch, i, f"outputs/analysis/real/img/batch_{i}")
    
    all_params_td_real.extend(batch_params_td)
    all_params_freq_real.extend(batch_params_freq)
    all_nonn_time_real.extend(batch_noNNTime)
    # all_hr_dataset.extend([dictionary.hr for dictionary in batch_params_td])
    # all_rmssd_dataset.extend([dictionary.rmssd for dictionary in batch_params_td])
    # all_lf_to_hf_dataset.extend([dictionary.LF_TO_HF for dictionary in batch_params_freq])
    # all_nonntime_dataset.extend(batch_noNNTime)
    
    
    #out_file = open(f"outputs/analysis/fake/json/batch_{i}.json", "x")
    #json.dump(param_dict, out_file, indent=6)
    #out_file.close()

all_hr_real = [obj.hr for obj in all_params_td_real]
all_rmssd_real = [obj.rmssd for obj in all_params_td_real]
all_lf_to_hf_real = [obj.LF_TO_HF for obj in all_params_freq_real]
all_nonn_time_real = all_nonn_time_real
all_lf_to_p_real = [obj.LF_TO_P for obj in all_params_freq_real]
all_hf_to_p_real = [obj.HF_TO_P for obj in all_params_freq_real]
all_vlf_to_p_real = [obj.VLF_TO_P for obj in all_params_freq_real]
all_shannon_real = [obj.shannon for obj in all_params_td_real]

    
json_output = param_dict_to_json(all_params_td_real, all_params_freq_real, all_nonn_time_real)
out_file = open(f"outputs/analysis/real/stats.json", "x")
json.dump(json_output, out_file, indent=6)
out_file.close()
plot_param_hist(all_hr_real, "Mean heart rate of 320s ECGs", "Heart Rate [BPM]", "Relative frequency", "real_hr.png")
plot_param_hist(all_rmssd_real, "Rmssd of 320s ECGs", "rmssd [ms]", "Relative frequency", "real_rmssd.png")
plot_param_hist(all_lf_to_hf_real, "LF to HF ratio of 320s ECGs", "LF/HF" , "Relative frequency", "real_lf_to_hf.png")
plot_param_hist(all_nonn_time_real, "Filter activity of  320s ECGs", "NoNN time [ms]", "Relative frequency", "real_nonntime.png")
plot_param_hist(all_lf_to_p_real,  "LF to P ratio of  320s ECGs","LF / P", "Relative frequency", "real_lf_to_p.png")
plot_param_hist(all_hf_to_p_real, "HF to P ratio of  320s ECGs", "HF / P", "Relative frequency", "real_Hf_to_p.png")
plot_param_hist(all_vlf_to_p_real, "VLF to P ratio of  320s ECGs" , "VLF / P", "Relative frequency", "real_vlf_to_p.png")
plot_param_hist(all_shannon_real, "Shannon Entropy of 320s ECGs", "Shannon Entropy", "Relative Frequency", "real_shannon.png")





#! Mann whitney u tests

hr = mannwhitneyu(all_hr_real, all_hr_fake)
rmssd = mannwhitneyu(all_rmssd_real, all_rmssd_fake)
lf_to_hf = mannwhitneyu(all_lf_to_hf_real, all_lf_to_hf_fake)
nonntime = mannwhitneyu(all_nonn_time_real, all_nonntime_fake)
vlf = mannwhitneyu(all_vlf_to_p_real, all_vlf_to_p_fake)
hf = mannwhitneyu(all_hf_to_p_real, all_hf_to_p_fake)
lf = mannwhitneyu(all_lf_to_p_real, all_lf_to_p_fake)
shannon = mannwhitneyu(all_shannon_real, all_shannon_fake)


test_output = {"hr": hr, "rmssd": rmssd, "lf_to_hf": lf_to_hf, "nonntime": nonntime, "vlf_to_p": vlf,
               "hf_to_p": hf, "lf_to_p": lf, "shannon": shannon
               }

test_out_file = open(f"outputs/analysis/test.json", 'x')
json.dump(test_output, test_out_file, indent=6)
test_out_file.close()

print(hr)
print(rmssd)
print(lf_to_hf)
print(nonntime)
print(vlf)
print(hf)
print(lf)