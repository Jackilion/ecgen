from typing import Iterator
import pandas
import random
import os
import numpy
import jax.numpy as jnp
from config.config import Config

#! TODO: Make it more memeory efficient, maybe send multiple chunks at once to the gpu. Right now this saves only about 15 A100 minutes compared to the segment by segment version
def data_loader(num_batches: int, data_path: str) -> Iterator:
    config = Config().settings
    for i in range(1, num_batches + 1):
        #open chunk i and load all files there into GPU memory
        print("\nloading chunk " + str(i))

        #!This saves about 15 A100 minutes, but there is a nan error in there somewhere, in the normalisation. I don't understand why, cause as I see it, the normalisation is the same as in normal numpy...
        #! The division is the problem, so maybe there are some zero segments in the data? But I don't understand why the min-max normalisation works in the CPU version and not in the GPU version...
        # ecg_blocks = []
        # for file in os.listdir(data_path + str(i)):
        #     if not file.endswith(".parquet"):
        #         continue
        #     data=pandas.read_parquet(data_path + str(i) + "/" + file)
        #     block = data["ecg_filtered"]
        #     #take first 8 segments
        #     for j in range(8):
        #         length = len(block[j])
        #         if not length % config["sample_length"] == 0:
        #             continue
        #         ecg_blocks.append(block[j])
            
        #     #quit()
        # stacked = numpy.stack(ecg_blocks)
        # #transfer to GPU
        # gpu_stacked = jnp.array(stacked)

        # B, L = gpu_stacked.shape
        # gpu_reshaped = gpu_stacked.reshape((B, 160, config["sample_length"])) #(B, 160, 1024)
        # print(jnp.isnan(gpu_reshaped).any())
        # gpu_normalised = (gpu_reshaped - jnp.min(gpu_reshaped, axis=2, keepdims=True)) / (jnp.max(gpu_reshaped, axis=2, keepdims=True) - jnp.min(gpu_reshaped, axis=2, keepdims=True))
        # print(jnp.isnan(gpu_normalised).any())
        # lower_part = jnp.max(gpu_reshaped, axis=2, keepdims=True) - jnp.min(gpu_reshaped, axis=2, keepdims=True)
        # print(jnp.isnan(lower_part).any())
        # upper_part = (gpu_reshaped - jnp.min(gpu_reshaped, axis=2, keepdims=True))
        # print(jnp.isnan(upper_part).any())
        # print(jnp.isnan(upper_part / lower_part).any())
        # quit()
        # for i in range(B):
        #     yield gpu_normalised[i]
        # #just to be sure...
        # del stacked
        # del gpu_stacked
        # del gpu_reshaped







        for file in os.listdir(data_path  + str(i)):
            if not file.endswith(".parquet"):
                continue
            
            data = pandas.read_parquet(data_path + str(i) + "/" + file)
            ecg_block = data["ecg_filtered"] #Ecg_block has 128 320s segments sampled at 512 Hz
            #segments are shuffled, so to "draw" 8 blocks out of the 128, we just take the first 8
            # whole chunk with 1000 patients would already be 133 million training samples
            # so we take just 8 segments for now, resulting in 8 million samples
            for j in range(8):
                length = len(ecg_block[j])
                if not length % config["AE_sample_length"] == 0:
                    continue
                reshaped = numpy.reshape(ecg_block[j], (-1, config["AE_sample_length"]))

                normalised = (reshaped - numpy.min(reshaped, axis=1, keepdims=True)) / (numpy.max(reshaped, axis=1, keepdims=True) - numpy.min(reshaped, axis=1, keepdims=True))
                #reshape into 1024 sample segments
                #print(ecg_block[j].shape)
                #quit()
                #reshaped = numpy.reshape(ecg_block[j], (-1, config["sample_length"])) # (160, 1024)
                gpu = jnp.array(normalised)
                # print(gpu.shape)
                # quit()
                yield gpu


def load_tokenised_dataset(data_path):
    for file in os.listdir(data_path):
        if not file.endswith(".npz"):
            continue
        data = numpy.load(data_path + file)
        batch = data.files[0:64]
        batch = jnp.array([data[batch[i]] for i in range(64)])
        # print("Yielding batch")
        yield batch
        #for i in range(len(data.files)):
        #    print(data[data.files[i]].shape)
        #    quit()
        #    yield data[data.files[i]]
                

def load_afib_dataset(data_path):
    segments = numpy.load(data_path + "train_segments.npy")
    labels = numpy.load(data_path + "train_labels.npy")
    
    return segments, labels

def load_physionet_afib_dataset(data_path):
    train_segments = numpy.load('/home/dominik.kranz/data/ecg/physionet_afib_challenge/npy/train_segments.npy')
    train_labels = numpy.load('/home/dominik.kranz/data/ecg/physionet_afib_challenge/npy/train_labels_onehot.npy')
    test_segments = numpy.load('/home/dominik.kranz/data/ecg/physionet_afib_challenge/npy/test_segments.npy')
    test_labels = numpy.load('/home/dominik.kranz/data/ecg/physionet_afib_challenge/npy/test_labels_onehot.npy')

    #remove every segment and label [0, 0, 1, 0]
    remove_indices = []
    for i in range(len(train_labels)):
        if train_labels[i][2] == 1:
            remove_indices.append(i)
    train_segments = numpy.delete(train_segments, remove_indices, axis=0)
    train_labels = numpy.delete(train_labels, remove_indices, axis=0)

    #same for test
    remove_indices = []
    for i in range(len(test_labels)):
        if test_labels[i][2] == 1:
            remove_indices.append(i)
    test_segments = numpy.delete(test_segments, remove_indices, axis=0)
    test_labels = numpy.delete(test_labels, remove_indices, axis=0)

    #labels are still of shape (B, 4), but we removed all [0, ,0, 1, 0] labels
    #reshape labels to (B, 3)
    train_labels = numpy.delete(train_labels, 2, axis=1)
    test_labels = numpy.delete(test_labels, 2, axis=1)

    print(train_segments.shape)
    print(train_labels.shape)
    print(test_labels.shape)

    
    #cut into batches of 16 (or less for the last one)
    train_segments = jnp.array(train_segments)
    train_segments = jnp.array_split(train_segments, int(numpy.ceil(train_segments.shape[0] / 16)))
    train_labels = jnp.array(train_labels)
    train_labels = jnp.array_split(train_labels, int(numpy.ceil(train_labels.shape[0] / 16)))

    test_segments = jnp.array(test_segments)
    test_segments = jnp.array_split(test_segments, int(numpy.ceil(test_segments.shape[0] / 16)))
    test_labels = jnp.array(test_labels)
    test_labels = jnp.array_split(test_labels, int(numpy.ceil(test_labels.shape[0] / 16)))
    return train_segments, train_labels, test_segments, test_labels

    # test_segments = onp.load('/home/dominik.kranz/data/ecg/physionet_afib_challenge/npy/test_segments.npy')
    # test_labels = onp.load('/home/dominik.kranz/data/ecg/physionet_afib_challenge/npy/test_labels_onehot.npy')
