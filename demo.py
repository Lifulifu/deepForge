#%%
import os
import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model

from util import load_mnist, onehot
#%%
def plot_table(G, D, name, random=True):
    _, _, imgs, digits = load_mnist()
    input_imgs = [] # imgs of 0 ~ 9
    for i in range(10):
        imgs_filtered = imgs[np.where(digits == i)[0]]
        if random:
            idx = np.random.randint(0, imgs_filtered.shape[0])
        else:
            idx = 0
        input_imgs.append(imgs_filtered[idx])

    fig, axs = plt.subplots(10, 10)
    fig.set_size_inches(50, 50)
    for i in range(10): # for input img
        for j in range(10): # for target digit
            gen_img = G.predict([np.expand_dims(input_imgs[i], axis=0), onehot(np.full((1, 1), j), 10)])
            axs[i, j].imshow(gen_img[0, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
    plt.show()

    fig.savefig(name, dpi=150)

def plot_fig(G, G_mask, n, t):

    _, _, imgs, digits = load_mnist()

    target = onehot(np.full((1, 1), t), 10)
    use_img = imgs[n][np.newaxis, ...]
    gen_img = G.predict([use_img, target])
    mask_img = G_mask.predict([use_img, target])
    fig, axs = plt.subplots(1, 3, figsize=(4, 3))
    axs[0].imshow(use_img[0, :, :, 0], cmap='gray')
    axs[0].axis('off')
    axs[1].imshow(mask_img[0, :, :, 0], cmap='gray')
    axs[1].axis('off')
    axs[2].imshow(gen_img[0, :, :, 0], cmap='gray')
    axs[2].axis('off')
    fig.savefig(os.path.join(f'./tmp/fig_{n}_{t}'))

if '__main__' == __name__:

    version_name = 'test'
    # version_name = '16_inception_G2D1_model_50000iter'

    model_id = '50'
    G = load_model(f'./outputs/{version_name}/models/G{model_id}.hdf5')
    # G_mask = load_model(f'./outputs/{version_name}/models/G_mask{model_id}.hdf5')
    plot_table(G, None, f'./outputs/{version_name}/table_{model_id}.png')

    # for i in range(0, 60):
    #     plot_fig(G, G_mask, i, 8)
    #     plot_fig(G, G_mask, i, 9)
    #     plot_fig(G, G_mask, i, 3)

# %%
