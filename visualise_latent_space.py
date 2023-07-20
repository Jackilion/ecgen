import numpy as onp
import matplotlib.pyplot as plt
import sklearn.manifold

def meanNN_to_color(meanNN):
    hr = 1000 * 60 / meanNN
    maxHR = 160
    minHR = 130
    
    relative = hr/maxHR
    relative = onp.clip([relative], 0.0, 1.0)
    
    print(hr)
    print(relative)
    print(meanNN)
    
    # relative = relative * 255
    return relative[0], relative[0], relative[0]
    
    
    

latent_space = onp.load("latent_space_layer_norm.npy")
labels = onp.load("labels_layer_norm.npy")

print(latent_space.shape)
print(labels.shape) #B

latent_space = latent_space.reshape((288 * 64, 64 * 64))
labels = labels.reshape((288 * 64, 3))



embedded = sklearn.manifold.TSNE(n_components=2, learning_rate="auto", init="random", perplexity=30).fit_transform(latent_space)
x = embedded[0:500, 0]
y = embedded[0:500, 1]
c= labels[0:500, 0]
plt.scatter(x, y, c=c)
plt.savefig("latent_space_meanNN.png")
plt.close()

x = embedded[0:500, 0]
y = embedded[0:500, 1]
c= labels[0:500, 1]
plt.scatter(x, y, c=c)
plt.savefig("latent_space_sdNN.png")
plt.close()

x = embedded[0:500, 0]
y = embedded[0:500, 1]
c= labels[0:500, 2]
plt.scatter(x, y, c=c)
plt.savefig("latent_space_rmssd.png")
plt.close()


