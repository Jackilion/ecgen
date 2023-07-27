import numpy as onp
import jax

def load_ecg_dataset(rng, series_length, batch_size):
    data = onp.load("data/ecgs_1024.npy")
    labels = onp.load("data/labels_1024.npy")

    print(data.shape)
    print(labels.shape)
    
    P, M, L = labels.shape #Patients, minutes, labels
    P, M, S = data.shape #Patients, minutes, samples
    # data_reshaped = data.reshape((P,M*S))
    # labels_reshaped = labels.reshape((P, M*L))
    # labels_repeat = 

    data_2d = onp.reshape(data, (P*M, S))
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

def load_latent_space_dataset():
    return -1