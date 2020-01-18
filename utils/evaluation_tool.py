from math import ceil

from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

from .util import load_mnist, onehot


class Evaluator:
    def __init__(self, gen_model, gen_mask_model=None, mnist_model='./outputs/D_digit.hdf5'):
        self.G = load_model(gen_model)
        self.clf = load_model(mnist_model)

    def run(self, imgs, targets):
        self.imgs, self.targets = imgs, targets
        onehot_targets = onehot(np.array(targets), 10)
        self.adds = self.G.predict([imgs, onehot_targets])

    def classify(self, imgs):
        if imgs.ndim < 4:
            imgs = imgs[np.newaxis, :]
        clf_prob = self.clf.predict(imgs)
        clf_res = np.argmax(clf_prob, axis=1)
        return clf_res

    def score(self):
        clf_res = self.classify(self.adds)
        return {
            'samples': len(clf_res),
            'hit_rate': np.mean(clf_res==self.targets),
            'hit_amount': np.sum(clf_res==self.targets)
        }

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
    def __init__(self, data='test'):
        if data == 'test':
            _, _, self.imgs, self.digits = load_mnist()
        else:
            self.imgs, self.digits, _, _ = load_mnist()
        self.set = {i: [] for i in range(10)}
        for img, digit in zip(self.imgs, self.digits):
            self.set[digit].append(img)

    def sample(self, batch_size=10, idx=None):
        idx = np.array( list(range(batch_size)) if idx is None else idx )
        return self.imgs[idx], self.digits[idx]

    def sample_by_key(self, key, batch_size=10, idx=None):
        idx = np.array( list(range(batch_size)) if idx is None else idx )
        return self.set[key][idx], [key]*batch_size


def acc(x, y):
    return np.sum(x==y, axis=1)

def evaluate_model(G):
    s = Sampler()
    e = Evaluator(G)
    hit_amount, hit_rate = np.zeros((10, 10)), np.zeros((10, 10))
    for input_number in range(10):
        for target_number in range(10):
            imgs = np.array(s.set[input_number])
            e.run(imgs, [target_number]*len(imgs))
            score = e.score()
            hit_amount[input_number, target_number] = score['hit_amount']
            hit_rate[input_number, target_number] = score['hit_rate']
    total_score = np.sum(hit_amount)
    for i in range(10): # substract hits on the same number
        total_score -= hit_amount[i, i]
    return hit_amount, hit_rate, total_score

def plot_matrix(m, save_dir=None):
    fig, ax = plt.subplots()
    ax.matshow(m, cmap=plt.cm.Blues)
    ax.xaxis.set_ticks(np.arange(0, 10, 1))
    ax.yaxis.set_ticks(np.arange(0, 10, 1))
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            color = 'white' if m[i, j] > .5 else 'black'
            ax.text(j, i, f'{m[i,j]:.2f}'.lstrip('0'), va='center', ha='center', color=color)
    if save_dir:
        fig.savefig(save_dir)


if '__main__' == __name__:

    batch_size = 20
    e = Evaluator('./outputs/19_inception_G1D10_model_100000iter/models/G40000.hdf5')
    s = Sampler(batch_size)
    imgs, labels = s.sample_by_key(key=1)

    c.add_imgs(imgs, labels, [4]*len(imgs))
    res = c.score()
    print(res)

