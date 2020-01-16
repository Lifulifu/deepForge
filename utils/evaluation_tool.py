from math import ceil

from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

from .util import load_mnist, onehot


class Evaluator:
    def __init__(self, gen_model, gen_mask_model, mnist_model='./outputs/D_digit.hdf5'):
        self.G = load_model(gen_model)
        self.G_mask = load_model(gen_mask_model)
        self.clf = load_model(mnist_model)

    def add_imgs(self, imgs, digits, targets):
        self.imgs, self.digits, self.targets = imgs, digits, targets
        onehot_targets = onehot(np.array(targets), 10)
        self.masks = self.G_mask.predict([imgs, onehot_targets])
        self.adds = self.G.predict([imgs, onehot_targets])

    def classify(self, imgs):
        if imgs.ndim < 4:
            imgs = imgs[np.newaxis, :]
        clf_prob = self.clf.predict(imgs)
        clf_res = np.argmax(clf_prob, axis=1)
        return clf_res

    def score(self):
        clf_res = self.classify(self.adds)
        return np.mean(clf_res==self.targets)

    def _plot_fig(self, img, mask, add, text=None):
        fig, axs = plt.subplots(1, 3, figsize=(4, 3))
        axs[0].imshow(img[:, :, 0], cmap='gray')
        axs[0].axis('off')
        axs[1].imshow(mask[:, :, 0], cmap='gray')
        axs[1].axis('off')
        axs[2].imshow(add[:, :, 0], cmap='gray')
        axs[2].axis('off')
        if text is not None:
            axs[2].title.set_text(text)
        fig.show()
        return fig

    def plot_all(self):
        for img, mask, add, target in zip(self.imgs, self.masks, self.adds, self.targets):
            clf_res = self.classify(add)
            text = f'fail as {clf_res}' if clf_res != target else 'success'
            self._plot_fig(img, mask, add, text)


class Sampler:
    def __init__(self, batch_size):
        _, _, self.imgs, self.digits = load_mnist()
        self.set = {i: [] for i in range(10)}
        self.batch_size = batch_size
        for img, digit in zip(self.imgs, self.digits):
            self.set[digit].append(img)
        # for k, v in self.set.items():
        #     print(k, len(v))

    def sample(self, idx=None):
        n_batch = self.n_batch()
        idx = np.random.randint(0, n_batch) if idx is None else idx
        s = self.batch_size * idx
        e = s + self.batch_size
        return self.imgs[s:e], self.digits[s:e]

    def sample_by_key(self, key=None, idx=None):
        key = np.random.randint(0, 10) if key is None else key
        n_batch = self.n_batch(key)
        idx = np.random.randint(0, n_batch) if idx is None else idx
        s = self.batch_size * idx
        e = s + self.batch_size
        return self.set[key][s:e], [key]*self.batch_size

    def n_batch(self, key=None):
        if key is None:
            n_samples = len(self.imgs)
        else:
            n_samples = len(self.set[key])
        return ceil(n_samples/self.batch_size)


def acc(x, y):
    return np.sum(x==y, axis=1)

if '__main__' == __name__:

    batch_size = 20
    c = Evaluator('./outputs/19_inception_G1D10_model_100000iter/models/G50000.hdf5',
                  './outputs/19_inception_G1D10_model_100000iter/models/G_mask50000.hdf5')
    s = Sampler(batch_size)
    imgs, labels = s.sample_by_key(key=1)

    c.add_imgs(imgs, labels, [4]*len(imgs))
    res = c.score()
    print(res)
