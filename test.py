import numpy as onp
import matplotlib.pyplot as plt

data = onp.load("latent_space_sigmoid.npy")

ecg = data[0, 0]
print(ecg.shape)
plt.plot(ecg.flatten())
plt.savefig("test4.png")
plt.figure()
plt.plot(ecg.flatten()[1000:2000])
plt.savefig("test5.png")
plt.close()