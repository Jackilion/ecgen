import numpy
from dataclasses import dataclass
from scipy.integrate import trapezoid
from scipy.signal import welch
from scipy.stats import entropy

@dataclass
class FrequencyDomainParameters:
    P: float
    HF: float
    LF: float
    VLF: float
    ULF: float
    HF_TO_P: float
    LF_TO_P: float
    VLF_TO_P: float
    ULF_TO_P: float
    LF_TO_HF: float
    
@dataclass
class TimeDomainParameters:
    hr: float
    meanNN: float
    sdNN: float
    rmssd: float
    # cvNN: float
    # sdaNN1: float
    # pNN50: float
    # pNNI20: float
    # renyi4: float
    shannon: float
    
    

def calculate_time_domain_parameters(hrv_series: numpy.array) -> dict[str, float]:
    '''
        Calculates time domain parameters and returns them as a dict<string, number>.
        Parameters calculated:
            * meanNN
            * hr
            * sdNN
            * rmssd
            * shannon
    '''
    meanNN = numpy.mean(hrv_series)
    hr = 1000 * 60 / meanNN
    sdNN = numpy.std(hrv_series)
    diff = numpy.diff(hrv_series)
    diff_squared = diff ** 2
    rmssd = numpy.sqrt(numpy.sum(diff_squared) /len(diff_squared) )
    
    #shannon:
    [hist, bin_edges] = numpy.histogram(hrv_series, bins=100)
    hist_sum = sum(hist)
    probs = [x / hist_sum for x in hist]   
    shannon = entropy(probs)

    return TimeDomainParameters(
        hr=hr,
        meanNN=meanNN,
        sdNN=sdNN,
        rmssd=rmssd,
        shannon=shannon      
    )
    
    
    
def calculate_frequency_domain_parameters(hrv_slice: numpy.array) -> dict[str, float]:
    
    #interpolate:
    x = numpy.cumsum(hrv_slice)
    x_new = numpy.arange(start=0, stop=int(x[-1]), step=1000)

    x = x * 0.01
    y = hrv_slice
    x_new = x_new * 0.01
    
    interpolated = numpy.interp(x_new, x, y)
    
    frequencies, power = welch(interpolated)
    
    #Calc Frequency bands:
    lf_ind = numpy.where(numpy.logical_and(frequencies > 0.04, frequencies < 0.15))
    lf_signal = power[lf_ind]
    lf_freq = frequencies[lf_ind]
    
    hf_ind = numpy.where(numpy.logical_and(frequencies > 0.15, frequencies < 0.4))
    hf_signal = power[hf_ind]
    hf_freq = frequencies[hf_ind]
    
    ulf_ind = numpy.where(numpy.logical_and(frequencies > 0.0, frequencies < 0.003))
    ulf_signal = power[ulf_ind]
    ulf_freq = frequencies[ulf_ind]
    
    vlf_ind = numpy.where(numpy.logical_and(frequencies > 0.003, frequencies < 0.04))
    vlf_signal = power[vlf_ind]
    vlf_freq = frequencies[vlf_ind]
    
    p_ind = numpy.where(numpy.logical_and(frequencies > 0.0, frequencies < 0.4))
    p_signal = power[p_ind]
    p_freq = frequencies[p_ind]
    
    #Calc integrals:
    ulf_power = trapezoid(ulf_signal, ulf_freq)
    vlf_power = trapezoid(vlf_signal, vlf_freq)
    lf_power = trapezoid(lf_signal, lf_freq)
    hf_power = trapezoid(hf_signal, hf_freq)
    p_power = trapezoid(p_signal, p_freq)
    
    return FrequencyDomainParameters(
        P=p_power,
        HF=hf_power,
        LF=lf_power,
        VLF=vlf_power,
        ULF=ulf_power,
        HF_TO_P=hf_power / p_power,
        LF_TO_P=lf_power / p_power,
        VLF_TO_P=vlf_power / p_power,
        ULF_TO_P=ulf_power / p_power,
        LF_TO_HF=lf_power / hf_power
    )