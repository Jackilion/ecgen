import numpy as onp
import jax

def load_ecg_dataset(rng, series_length, batch_size):
    data = onp.load("data/ecgs_512.npy")
    labels = onp.load("data/labels_512.npy")

    print(data.shape)

    data_2d = onp.reshape(data, (-1, series_length))
    shuffled_indices = jax.random.permutation(rng, data_2d.shape[0])
    randomized_data_2d = data_2d[shuffled_indices, :]

    labels_2d = onp.reshape(labels, (-1, 3))
    randomized_labels_2d = labels_2d[shuffled_indices, :]

    batched_data = onp.array_split(randomized_data_2d, 18432 // batch_size)
    batched_labels = onp.array_split(randomized_labels_2d, 18432 // batch_size)

    return (batched_data, batched_labels)

def load_latent_space_dataset():
    return -1