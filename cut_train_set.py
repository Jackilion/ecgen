import h5py
import numpy
import matplotlib.pyplot as plt
import scipy.signal
import tqdm
import hrv


def divide_chunks(l, n):
    """Yields successive n-sized chunks from series l

    Args:
        l (list): the series
        n (list): number of chunks yielded

    Yields:
        list: The chunk
    """

    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


def time_to_bbi_loc(bbi_series, time):
    sum = 0
    index = 0
    for bbi in bbi_series:
        sum += bbi
        index += 1
        if sum > time:
            return index
    return -1


def divide_bbi_chunks(bbi_series, chunk_length):
    num_of_loops = int(sum(bbi_series) / chunk_length)
    chunks = []
    rest_series = bbi_series

    for i in range(num_of_loops):
        cutoff_point = time_to_bbi_loc(rest_series, chunk_length)
        chunk = rest_series[0:cutoff_point]
        chunks.append(chunk)

        rest_series = rest_series[cutoff_point:len(rest_series)]
    return chunks


def cut(filename, sample_frequency):
    """
    Opens the dataset (.h5) and generates a new dataset
    with ECGs of the given lengths
    """

    file = h5py.File(filename, 'r')
    print(file.keys())
    print(file["ECG"])
    print(file["BBI"])
    print(file["BBI"][0])
    all_ecgs = file["ECG"][:]

    # print(all_ecgs.shape)
    # print(all_ecgs[0].shape)
    # print(file["BBI"].shape)
    # print(file["RP"].shape)
    # print(file["RP"][0][0:1000])
    # print(file["RP"][0][1000:2000])
    ecg = all_ecgs[0][0:10000]
    labels = file["RP"][0][0:10000]
    labels = [x if x == 3.0 else 0 for x in labels]
    plt.plot(ecg)
    plt.ylim(-0.003, 0.01)
    plt.plot(labels, alpha=0.2)
    plt.savefig("test.png")
    # print(labels)
    rpeak_count = numpy.count_nonzero(labels == 3.0)

    print(rpeak_count)

    dataset = numpy.ndarray(shape=(1024, 18, 60 * sample_frequency))
    labels = numpy.ndarray(shape=(1024, 18, 3))

    pbar = tqdm.tqdm(range(1024))
    for i in pbar:
        eighteen_minutes_ecg = all_ecgs[i]
        eighteen_minutes_rpeaks = file["RP"][i]
        full_bbi = file["BBI"][i]

        one_minute_bbi = divide_bbi_chunks(full_bbi, 60*1000)

        one_minute_chunks = numpy.array(list(divide_chunks(
            eighteen_minutes_ecg, 60 * 1024)))
        one_minute_chunks_resampled = scipy.signal.resample(
            one_minute_chunks, num=60 * sample_frequency, axis=-1)

        for j in range(len(one_minute_chunks_resampled)):
            one_minute = one_minute_chunks_resampled[j]

            r_peaks = scipy.signal.find_peaks(
                one_minute, 0.005, None, 200)
            r_peaks_loc = r_peaks[0]
            r_peaks_heights = r_peaks[1]["peak_heights"]
            avg_peak_height = numpy.average(r_peaks_heights)

            normalized_one_minute = one_minute / avg_peak_height
            r_peaks_time = r_peaks_loc / sample_frequency

            hrv_series = numpy.diff(r_peaks_time) * 1000
            time_domain_params = hrv.calculate_time_domain_parameters(
                hrv_series)
            dataset[i, j] = normalized_one_minute
            labels[i, j] = [time_domain_params["meanNN"],
                            time_domain_params["sdNN"], time_domain_params["rmssd"]]

        # print(time_domain_params["meanNN"])
        # print(time_domain_params["hr"])
        # print(time_domain_params["sdNN"])
        # print(time_domain_params["rmssd"])
        # print("--------------------------")

        # plt.figure()
        # plt.plot(normalized_one_minute)
        # plt.plot(r_peaks_loc, [normalized_one_minute[x]
        #          for x in r_peaks_loc], linestyle="", marker="x")
        # plt.savefig("one_min_ecg_annotated.png")
        # plt.close()
    numpy.save("dataset_512.npy", dataset)
    numpy.save("labels_512.npy", labels)


cut("../data/synthetic/ecgsyn.h5", 512)
