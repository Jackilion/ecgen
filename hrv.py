import numpy as np


def calculate_time_domain_parameters(hrv_series: np.array) -> dict[str, float]:
    '''
        Calculates time domain parameters and returns them as a dict<string, number>.
        Parameters calculated:
            * meanNN
            * hr
            * sdNN
            * rmssd
    '''
    meanNN = np.mean(hrv_series)
    hr = 1000 * 60 / meanNN
    sdNN = np.std(hrv_series)
    diff = np.diff(hrv_series)
    diff_squared = diff ** 2
    rmssd = np.sqrt(np.sum(diff_squared) / len(diff_squared))

    return {
        "meanNN": meanNN,
        "hr": hr,
        "sdNN": sdNN,
        "rmssd": rmssd
    }
