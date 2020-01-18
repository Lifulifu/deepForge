from utils.evaluation_tool import Evaluator, Sampler, evaluate_model, plot_matrix
from utils.util import load_mnist, plot_table
import numpy as np
from keras.models import load_model

models = [
    './outputs/18_inception_G2D1_model_100000iter/models/G100000.hdf5',
]
scores = []
for model in models:
    hit_amount, hit_rate, score = evaluate_model(model)
    scores.append(score)
    plot_matrix(hit_rate, save_dir='./matrix.png')
    plot_table(model, 'mytable.png', save=True)
print(scores)



