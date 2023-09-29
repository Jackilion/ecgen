import numpy as np

# Helper Function to Calculate Adaptive Moments
def calculate_adaptive_moments(timeseries, initialization_length, CONFIG):
    adaptive_mean = np.nan * np.ones_like(timeseries)
    adaptive_variance = np.nan * np.ones_like(timeseries)
    # Initialize...
    adaptive_initialization_idxs = range(initialization_length)
    adaptive_mean[0] = np.mean(timeseries[adaptive_initialization_idxs])
    adaptive_variance[0] = np.mean((adaptive_mean[0] - timeseries[adaptive_initialization_idxs]) ** 2)

    for jj in range(1, len(timeseries)):
        adaptive_mean[jj] = adaptive_mean[jj-1] - \
            CONFIG["adaptivity_controlling_coefficient"] * (adaptive_mean[jj-1] - timeseries[jj-1])
        
        last_variance_item = (adaptive_mean[jj-1] - timeseries[jj-1]) ** 2
        adaptive_variance[jj] = adaptive_variance[jj-1] - \
            CONFIG["adaptivity_controlling_coefficient"] * (adaptive_variance[jj-1] - last_variance_item)
    
    adaptive_mean = adaptive_mean.reshape(-1, 1)
    adaptive_sigma = np.sqrt(adaptive_variance).reshape(-1, 1)
#   plt.figure(); plt.subplot(2,1,1); plt.plot(adaptive_sigma, 'g'); plt.subplot(2,1,2); plt.plot(adaptive_mean, 'g'); plt.hold(True); plt.plot(timeseries, 'k')
    return adaptive_mean.reshape((-1)), adaptive_sigma.reshape((-1))




def adaptive_hrv_filter(hrv_timeseries, **kwargs):
    # Configuration
    # Default Values
    CONFIG = {}
    CONFIG["replace_nonnormal_values"] = True
    CONFIG["adaptivity_controlling_coefficient"] = 0.05
    CONFIG["range_proportionality_limit"] = 10 / 100
    CONFIG["outlier_min_z_factor"] = 3
    CONFIG["allowed_excess_hrv_variability"] = 20 # ms
    CONFIG["remove_nonphysiological_outliers"] = True # Helps with stability
    CONFIG["mimimum_physiological_value"] = 200 # ms
    CONFIG["maximum_physiological_value"] = 2000 # ms

    # Parse Configuration parameters
    for key, value in kwargs.items():
        if value is None:
            continue
        key = key.lower()
        if key == "replace_nonnormal_values":
            assert isinstance(value, bool), "adaptive_hrv_filter: parameter 'replace_nonnormal_values' requires a logical value"
            CONFIG["replace_nonnormal_values"] = value
        elif key == "remove_nonphysiological_outliers":
            assert isinstance(value, bool), "adaptive_hrv_filter: parameter 'remove_nonphysiological_outliers' requires a logical value"
            CONFIG["remove_nonphysiological_outliers"] = value
        else:
            raise ValueError("adaptive_hrv_filter: Unrecognized parameter-value pair specified.")

    # First Parameter Check
    if not (np.isscalar(hrv_timeseries) or np.ndim(hrv_timeseries) == 1) or not np.issubdtype(hrv_timeseries.dtype, np.number):
        raise ValueError("hrv_timeseries is not a numeric vector")

    if not (np.sum(np.size(hrv_timeseries) > 1) == 1):
        raise ValueError("hrv_timeseries is not a numeric vector with a single non-singular dimension.")

    hrv_timeseries = np.reshape(np.double(hrv_timeseries), (-1, 1))

    # Parameter Checks (just to be on the safe side)
    assert len(np.shape(hrv_timeseries)) == 2 and np.shape(hrv_timeseries)[1] == 1

    # Remove values that are out of range of Physiological Heart beats
    removed_nonphysiological_outliers = np.zeros_like(hrv_timeseries, dtype=bool)
    removed_nonphysiological_outliers_idx = []
    if CONFIG["remove_nonphysiological_outliers"]:
        #This is what the matlab code did, but it's not allowed in python, so we delete the items instead.
        removed_nonphysiological_outliers = (hrv_timeseries > CONFIG["maximum_physiological_value"]) | (hrv_timeseries < CONFIG["mimimum_physiological_value"])
        removed_nonphysiological_outliers_idx = np.where(removed_nonphysiological_outliers)[0]
        #hrv_timeseries[hrv_timeseries > CONFIG["maximum_physiological_value"]] = []
        #hrv_timeseries[hrv_timeseries < CONFIG["mimimum_physiological_value"]] = []
        hrv_timeseries = np.delete(hrv_timeseries, np.where(hrv_timeseries > CONFIG["maximum_physiological_value"]))
        hrv_timeseries = np.delete(hrv_timeseries, np.where(hrv_timeseries < CONFIG["mimimum_physiological_value"]))
    
    # Calculate helper matrices for the Binominal Filter
    bin_coeff = np.array([1, 6, 15, 20, 15, 6, 1])
    coeff_count = len(bin_coeff)
    coeff_sum = np.sum(bin_coeff)
    assert coeff_count % 2 == 1
    timeseries_length = len(hrv_timeseries)

    hrv_filter_value_mat = np.tile(bin_coeff, (timeseries_length, 1)).T

    value_index_column = np.reshape(np.arange(coeff_count) - np.floor(coeff_count / 2), (-1, 1))
    hrv_ind_offset_val = np.tile(value_index_column, (1, timeseries_length))
    hrv_ind_base_val = np.tile(np.arange(timeseries_length), (coeff_count, 1))

    # The first and last elements will be out of range...
    # ...work around by just (re-)using first and last elements...
    hrv_idx_mat = (hrv_ind_offset_val + hrv_ind_base_val).astype(int)
    hrv_idx_mat[hrv_idx_mat < 0] = 0
    hrv_idx_mat[hrv_idx_mat >= timeseries_length] = timeseries_length - 1

    # Calculate first filtered Signal (Through Binominal Filter)
    filtered_hrv = np.reshape(np.sum(hrv_timeseries[hrv_idx_mat] * hrv_filter_value_mat, axis=0) * (1 / coeff_sum), (-1, 1))
    
        # Apply "Adaptive Percent Filter" => fixed_hrv_timeseries
    adaptive_mean, adaptive_sigma = calculate_adaptive_moments(filtered_hrv, coeff_count, CONFIG)
    adaptive_sigma_mean = np.mean(adaptive_sigma)

    last_good_value = hrv_timeseries[0] # Maybe filtered_hrv[0]
    last_good_range = \
        CONFIG["range_proportionality_limit"] * last_good_value + \
        CONFIG["outlier_min_z_factor"] * adaptive_sigma_mean

    hrv_diff = np.diff(hrv_timeseries)
    normal_hrv_values = np.ones_like(filtered_hrv, dtype=bool)
    for ii in range(1, len(hrv_timeseries)):
        current_diff = abs(hrv_diff[ii-1])
        current_max_range = \
            CONFIG["range_proportionality_limit"] * hrv_timeseries[ii-1] + \
            CONFIG["outlier_min_z_factor"] * adaptive_sigma_mean

        current_value_is_normal = current_diff <= current_max_range or current_diff <= last_good_range
        if current_value_is_normal:
            last_good_value = hrv_timeseries[ii]
            last_good_range = \
                CONFIG["range_proportionality_limit"] * last_good_value + \
                CONFIG["outlier_min_z_factor"] * adaptive_sigma_mean
        normal_hrv_values[ii] = current_value_is_normal

    fixed_hrv_timeseries = hrv_timeseries.copy()
    non_normal_hrv_idxs = np.where(~normal_hrv_values)[0]
    
    adaptive_mean = adaptive_mean.reshape((-1))
    adaptive_sigma = adaptive_sigma.reshape((-1))
    fixed_hrv_timeseries[non_normal_hrv_idxs] = adaptive_mean[non_normal_hrv_idxs] + np.multiply((np.random.rand(len(non_normal_hrv_idxs)) - 0.5), adaptive_sigma[non_normal_hrv_idxs])
    #return fixed_hrv_timeseries
    
    # Calculate second filtered Signal (Through Binominal Filter)
    filtered_fixed_timeseries = np.reshape(np.sum(fixed_hrv_timeseries[hrv_idx_mat] * hrv_filter_value_mat, axis=0) * (1 / coeff_sum), (-1, 1))

    
    # Apply "Adaptive Controlling Procedure" => fixed_fixed_hrv_timeseries
    adaptive_fixed_mean, adaptive_fixed_sigma = calculate_adaptive_moments(filtered_fixed_timeseries, coeff_count, CONFIG)

    normal_fixed_hrv_values = \
        abs(fixed_hrv_timeseries - adaptive_fixed_mean) <= \
        (CONFIG["outlier_min_z_factor"] * adaptive_fixed_sigma + \
        CONFIG["allowed_excess_hrv_variability"])

    fixed_fixed_hrv_timeseries = fixed_hrv_timeseries.copy()
    non_normal_fixed_hrv_idxs = np.where(~normal_fixed_hrv_values)[0]
    fixed_fixed_hrv_timeseries[non_normal_fixed_hrv_idxs] = \
        adaptive_fixed_mean[non_normal_fixed_hrv_idxs] + \
        (np.random.rand(len(non_normal_fixed_hrv_idxs)) - 0.5) * adaptive_fixed_sigma[non_normal_fixed_hrv_idxs]

    
    # Set Return Values
    non_normal_hrv_idxs = np.where((~normal_fixed_hrv_values) | (~normal_hrv_values))[0]
    return_dict = {}
    if CONFIG["replace_nonnormal_values"]:
        # Return a fixed time series, a list of indices (relative to the returned list) that were changed to
        # create such time series, and a list of indices that were removed
        # from the input time series, because they were so out of range that
        # they would mess with the filters.
        return_dict = {}
        return_dict["filtered_hrv"] = fixed_fixed_hrv_timeseries
        #varargout = [fixed_fixed_hrv_timeseries]
        #if nargout > 1:
            #varargout.append(non_normal_hrv_idxs)
        #if nargout > 2:
            #varargout.append(removed_nonphysiological_outliers_idx)
        return_dict["non_normal_hrv_idxs"] = non_normal_hrv_idxs
        return_dict["removed_nonphysiological_outliers_idxs"] = removed_nonphysiological_outliers_idx
    else:
        # Return a list of indices relative to the input data, of data
        # points that were either removed, or changed toward normality.

        non_normal_input_data = removed_nonphysiological_outliers.copy()
        non_removed_data_points = np.setdiff1d(np.arange(len(non_normal_input_data)), removed_nonphysiological_outliers_idx)

        non_removed_non_normal_input_data = np.zeros_like(fixed_fixed_hrv_timeseries, dtype=bool)
        non_removed_non_normal_input_data[non_normal_hrv_idxs] = True

        non_normal_input_data[non_removed_data_points] = non_removed_non_normal_input_data
        
        
        
        #varargout = [np.where(non_normal_input_data)[0]]
        return_dict["filtered_hrv"] = np.where(non_normal_input_data)[0]

    return return_dict