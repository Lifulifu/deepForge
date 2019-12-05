#%%
import os
import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model

from util import load_mnist, onehot
#%%
def plot_table(G, D, name):
    _, _, imgs, digits = load_mnist()
    input_imgs = [] # imgs of 0 ~ 9
    for i in range(10):
        imgs_filtered = imgs[np.where(digits == i)[0]]
        idx = np.random.randint(0, imgs_filtered.shape[0])
        input_imgs.append(imgs_filtered[idx])

    fig, axs = plt.subplots(10, 10)
    fig.set_size_inches(60, 60)
    for i in range(10): # for input img
        for j in range(10): # for target digit
            gen_img = G.predict([np.expand_dims(input_imgs[i], axis=0), onehot(np.full((1, 1), j), 10)])
            axs[i, j].imshow(gen_img[0, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
    plt.show()
    fig.savefig(os.path.join(f'{name}.png'), dpi=150)

model_name = 'G10000'
G = load_model(f'models/14_inception_G10D5_model_10000iter/{model_name}.hdf5')
plot_table(G, None, model_name)

# %%
