from pathlib import Path
import sys
import numpy as onp
import jax
import pandas as pd
from tqdm import tqdm
import jax.numpy as jnp

def load_ecg_dataset(rng, series_length, batch_size, normalise=False, dataset_path="data/ecgs_1024.npy"):
    data = onp.load(dataset_path)
    labels = onp.load("data/labels_1024.npy")

    print(data.shape)
    print(labels.shape)
    
    P, M, L = labels.shape #Patients, minutes, labels
    P, M, S = data.shape #Patients, minutes, samples
    # data_reshaped = data.reshape((P,M*S))
    # labels_reshaped = labels.reshape((P, M*L))
    # labels_repeat = 

    data_2d = onp.reshape(data, (P*M, S))
    if normalise:
        max_val = onp.max(data_2d)
        min_val = onp.min(data_2d)
        range_vals = max_val - min_val
        data_2d = (data_2d + onp.abs(min_val)) / (max_val + onp.abs(min_val))
    shuffled_indices = jax.random.permutation(rng, data_2d.shape[0])
    randomized_data_2d = data_2d[shuffled_indices, :]

    labels_2d = onp.reshape(labels, (P*M, L))
    randomized_labels_2d = labels_2d[shuffled_indices, :]
    
    #Cut into desired length
    data_cut = onp.reshape(data_2d, (-1, series_length))
    data_size, data_length = data_cut.shape
    
    label_size, label_length = labels_2d.shape
    
    batched_data = onp.array_split(data_cut,  data_size // batch_size)
    batched_labels = onp.array_split(randomized_labels_2d, label_size // batch_size)
    print(len(batched_data))
    print(len(batched_labels))
    
    return (batched_data, batched_labels)

def load_afib_dataset_5s():
    location = "/home/dominik.kranz/data/ecg/inhouse_afib_dataset/"
    afib_5s = []
    print("Loading dataset")
    files = list(Path(location + "afib/").rglob("*.parquet"))
    
    count = 0
    for file in tqdm(files):
        if count > 1:
            break
        df = pd.read_parquet(file)
        ecg_30s = df["ecg_processed"].to_numpy()
        #print(onp.stack(ecg_30s).shape) #(10000, 15360)
        #reshape to 5s
        ecg_5s = onp.stack(ecg_30s).reshape((-1, 5*512))
        afib_5s.append(ecg_5s)
        count += 1
        
    
    afib_5s = onp.stack(afib_5s)
    afib_5s = onp.reshape(afib_5s, (-1, 5*512)) #flatten the first two dimensions => (B, 5*512)
    #min max normalise along the time axis
    afib_5s = (afib_5s - afib_5s.min(axis=1, keepdims=True)) / (afib_5s.max(axis=1, keepdims=True) - afib_5s.min(axis=1, keepdims=True) + 1e-6)
    #shuffle 
    print(afib_5s.shape)
    print(sys.getsizeof(afib_5s))

    #load sinus
    sinus_5s = []
    files = list(Path(location + "sinus/").rglob("*.parquet"))
    count = 0
    for file in tqdm(files):
        if count > 1:
            break
        df = pd.read_parquet(file)
        ecg_30s = df["ecg_processed"].to_numpy()
        ecg_5s = onp.stack(ecg_30s).reshape((-1, 5*512))
        sinus_5s.append(ecg_5s)
        count += 1
    sinus_5s = onp.stack(sinus_5s)
    sinus_5s = onp.reshape(sinus_5s, (-1, 5*512)) #flatten the first two dimensions => (B, 5*512)
    #min max normalise along the time axis
    sinus_5s = (sinus_5s - sinus_5s.min(axis=1, keepdims=True)) / (sinus_5s.max(axis=1, keepdims=True) - sinus_5s.min(axis=1, keepdims=True) + 1e-6)
    #shuffle
    
    #combine
    labels = onp.concatenate([onp.zeros(afib_5s.shape[0]), onp.ones(sinus_5s.shape[0])])
    whole = onp.concatenate([afib_5s, sinus_5s], axis=0)
    
    #print datatype
    print(whole.dtype)
    print(sys.getsizeof(whole))   
    #shuffle whole and labels
    indices = onp.arange(whole.shape[0])
    onp.random.shuffle(indices)
    whole = whole[indices]
    labels = labels[indices] 
    #take only every 5th sample
    whole = whole[::5]
    #send to GPU
    whole = jnp.array(whole)
    labels = jnp.array(labels)
    
    print(whole.shape)
    
    #split into list of chunks each of length 32
    chunked = jnp.array_split(whole, whole.shape[0] // 32)
    chunked_labels = jnp.array_split(labels, labels.shape[0] // 32)
    print("Dataset loaded")
    return chunked, chunked_labels

def load_afib_dataset_30s():
    location = "/home/dominik.kranz/data/ecg/inhouse_afib_dataset/"
    afib_30s = []
    print("Loading dataset")
    files = list(Path(location + "afib/").rglob("*.parquet"))
    
    count = 0
    for file in tqdm(files):
        # if count > 1:
        #     break
        df = pd.read_parquet(file)
        ecg_30s = df["ecg_processed"].to_numpy()
        #print(onp.stack(ecg_30s).shape) #(10000, 15360)
        #reshape to 5s
        # ecg_5s = onp.stack(ecg_30s).reshape((-1, 5*512))
        afib_30s.append(onp.stack(ecg_30s).reshape((-1, 30*512)))
        # print(onp.stack(ecg_30s).shape)
        count += 1
        
    
    afib_30s = onp.stack(afib_30s)
    afib_30s = onp.reshape(afib_30s, (-1, 30*512)) #flatten the first two dimensions => (B, 30*512)
    #min max normalise along the time axis
    afib_30s = (afib_30s - afib_30s.min(axis=1, keepdims=True)) / (afib_30s.max(axis=1, keepdims=True) - afib_30s.min(axis=1, keepdims=True) + 1e-6)
    #shuffle 
    print(afib_30s.shape)
    print(sys.getsizeof(afib_30s))

    #load sinus
    sinus_30s = []
    files = list(Path(location + "sinus/").rglob("*.parquet"))
    count = 0
    for file in tqdm(files):
        # if count > 1:
        #     break
        df = pd.read_parquet(file)
        ecg_30s = df["ecg_processed"].to_numpy()
        ecg_30s = onp.stack(ecg_30s).reshape((-1, 30*512))
        sinus_30s.append(ecg_30s)
        count += 1
    sinus_30s = onp.stack(sinus_30s)
    sinus_30s = onp.reshape(sinus_30s, (-1, 30*512)) #flatten the first two dimensions => (B, 5*512)
    #min max normalise along the time axis
    sinus_30s = (sinus_30s - sinus_30s.min(axis=1, keepdims=True)) / (sinus_30s.max(axis=1, keepdims=True) - sinus_30s.min(axis=1, keepdims=True) + 1e-6)
    #shuffle
    
    #combine
    labels = onp.concatenate([onp.zeros(afib_30s.shape[0]), onp.ones(sinus_30s.shape[0])])
    whole = onp.concatenate([afib_30s, sinus_30s], axis=0)
    
    #print datatype
    print(whole.dtype)
    print(sys.getsizeof(whole))   
    #shuffle whole and labels
    indices = onp.arange(whole.shape[0])
    onp.random.shuffle(indices)
    whole = whole[indices]
    labels = labels[indices] 
    #take only every 5th sample
    # whole = whole[::5]
    #send to GPU
    whole = jnp.array(whole)
    labels = jnp.array(labels)
    
    print(whole.shape)
    print(labels.shape)
    
    #split into list of chunks each of length 32
    chunked = jnp.array_split(whole, whole.shape[0] // 32)
    chunked_labels = jnp.array_split(labels, labels.shape[0] // 32)
    print("Dataset loaded")
    # quit()
    return chunked, chunked_labels

def load_latent_space_dataset():
    return -1