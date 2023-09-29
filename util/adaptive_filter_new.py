import numpy as np
import matplotlib.pyplot as plt


def filter_bin_7(series):
    bin_7_coef = (np.poly1d([1, 1])**6).coeffs / 64
    
    #pad the hrv with repeating edges (Similar to ANI2000)
    left = series[0]
    right = series[-1]
    padded = [left, left, left, *series, right, right, right]
    return np.convolve(bin_7_coef, padded, mode="valid")


def calculate_adaptive_moments(timeseries, c):
    #This is what Jan did in the matlab code.
    #I don't understand why it is identical to the paper
    #But it seems to work
    adaptive_mean = np.empty_like(timeseries)
    adaptive_variance = np.empty_like(timeseries)
    initialization_length = 7
    
    # Initialize...
    adaptive_initialization_idxs = np.arange(initialization_length)
    adaptive_mean[0] = np.mean(timeseries[adaptive_initialization_idxs])
    adaptive_variance[0] = np.mean((adaptive_mean[0] - timeseries[adaptive_initialization_idxs]) ** 2)
    
    for jj in range(1, len(timeseries)):
        adaptive_mean[jj] = adaptive_mean[jj-1] - \
            c * (adaptive_mean[jj-1] - timeseries[jj-1])
        
        last_variance_item = (adaptive_mean[jj-1] - timeseries[jj-1]) ** 2
        adaptive_variance[jj] = adaptive_variance[jj-1] - \
            c * (adaptive_variance[jj-1] - last_variance_item)
    
    adaptive_sigma = np.sqrt(adaptive_variance)
    
    return adaptive_mean, adaptive_sigma

def adaptive_moments(series, c):
    #! TODO: Check for bug
    adaptive_means = []
    adaptive_variances = []
    adaptive_second_moments = []
    
    adaptive_means.append(np.mean(series[0:7]))
    adaptive_second_moments.append(np.var( series[0:7]  ))
    adaptive_variances.append(adaptive_means[0]**2 - adaptive_second_moments[0])
    
    for i in range(1, len(series) + 1):
        
        adaptive_means.append(
            adaptive_means[i-1] - c * (adaptive_means[i-1] - series[i-1])
        )
        
        adaptive_second_moments.append(
            adaptive_second_moments[i-1] - c * (adaptive_second_moments[i-1] - series[i-1] ** 2)
        )
        
        adaptive_variances.append(adaptive_means[i]**2 - adaptive_second_moments[i])
    
    return adaptive_means, np.sqrt(adaptive_variances)

def adaptive_hrv_filter(hrv_timeseries):
    ADAPTIVITY_CONTROLLING_COEFFICIENT = 0.05
    RANGE_PROPORTIONALITY_LIMIT = 10 / 100
    OUTLIER_MIN_Z_FACTOR = 3.0
    BASIC_VARIABILITY = 20 #ms
    
    noNN_time = []
    
    #! 0. Remove nonphysiological outliers
    non_physiological_outliers = (hrv_timeseries < 200) | (hrv_timeseries > 2000)
    non_physiological_outliers_idxs = np.where(non_physiological_outliers)
    physiological_hrv = np.delete(hrv_timeseries, non_physiological_outliers_idxs)
    #! 1. Calculate the binomial-7-filter
    filtered_bin_7 = filter_bin_7(physiological_hrv)
    
    #! 2. Calc Adaptive Moments
    adap_mean, adap_std = calculate_adaptive_moments(filtered_bin_7, ADAPTIVITY_CONTROLLING_COEFFICIENT)
    mean_adap_std = np.mean(adap_std)
    #! 3. Filter
    adaptive_filtered_hrv = []
    adaptive_filtered_hrv.append(physiological_hrv[0])
    for i in range(1, len(physiological_hrv)):
        last_valid_bbi = physiological_hrv[0]
        
        #Check if beat is not normal:
        if (abs(physiological_hrv[i] - physiological_hrv[i-1])) > (RANGE_PROPORTIONALITY_LIMIT * physiological_hrv[i-1] + OUTLIER_MIN_Z_FACTOR * mean_adap_std)\
            and \
                abs(physiological_hrv[i] - last_valid_bbi) > (RANGE_PROPORTIONALITY_LIMIT * last_valid_bbi + OUTLIER_MIN_Z_FACTOR * mean_adap_std):
            noNN_time.append(physiological_hrv[i])
            adaptive_filtered_hrv.append(adap_mean[i] + (np.random.random() - 0.5) * adap_std[i])
            
        else:
            adaptive_filtered_hrv.append(physiological_hrv[i])
            last_valid_bbi = physiological_hrv[i]
                
    
    #! Calculate binomial-7-filter and moments again
    adap_filtered_bin_7 = filter_bin_7(adaptive_filtered_hrv)
    adap_mean, adap_std = calculate_adaptive_moments(adap_filtered_bin_7, ADAPTIVITY_CONTROLLING_COEFFICIENT)
    
    #! Adaptive controlling procedure
    fixed_adap_filtered_hrv = []
    fixed_adap_filtered_hrv.append(adaptive_filtered_hrv[0])
    for i in range(1, len(adaptive_filtered_hrv)):
        if abs(adaptive_filtered_hrv[i] - adap_mean[i]) > OUTLIER_MIN_Z_FACTOR * adap_std[i] + BASIC_VARIABILITY:
            noNN_time.append(adaptive_filtered_hrv[i])
            fixed_adap_filtered_hrv.append(adap_filtered_bin_7[i])
        else:
            fixed_adap_filtered_hrv.append(adaptive_filtered_hrv[i])
    
    return fixed_adap_filtered_hrv, sum(noNN_time)
    


# with open("util/002.txt") as f:
#     hrv = f.read().splitlines()
#     hrv = np.array(hrv, dtype=np.float32)
#     plt.plot(hrv)
#     plt.ylim([300, 1300])
    
#     plt.savefig("util/001.png")
#     filtered, noNN_time = adaptive_hrv_filter(hrv)
#     plt.figure()
#     plt.plot(filtered)
#     plt.ylim([300, 1300])
#     plt.savefig("util/001_filt.png")
#     plt.close()
#     print(len(hrv))
#     print(len(filtered))
#     print(hrv[0:10])
#     print(filtered[0:10])
#     for i in range(len(hrv)):
#         if(hrv[i] != filtered[i]):
#             print(f"index {i} is different")
#     print(noNN_time)
#     print(sum(noNN_time))