import argparse
import numpy as np
import pandas as pd
from multilayerPerceptron import *
import matplotlib.pyplot as plt

def test_train_split(df, ratio):
    rat = len(df) * ratio
    msk = np.random.rand(len(df)) < ratio
    x = df.loc[ : , df.columns != 'Diagnosis']
    y = df['Diagnosis']
    return x[msk], x[~msk], y[msk], y[~msk]

categories = [
    'radius',
    'texture',
    'perimeter',
    'area',
    'smoothness',
    'compactness',
    'concavity',
    'concave',
    'symmetry' ,
    'fractal dimension'
]

stats = [
    'mean',
    'standard error',
    'worst'
]

def add_columns_name(df):
    concat = ['ID', 'Diagnosis'] + \
        [' '.join([cat, stat]) for cat in categories for stat in stats ]
    df.columns = concat
    df.reset_index(drop=True, inplace=True)
    df.to_csv("./resources/data_p.csv", index=False)

import sys

def select_features(df):
    cor  = abs(df.corr())
    above_0 = cor[cor["Diagnosis"] > 0.2]
    return df[above_0.index]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, default="./resources/data.csv", help="dataset_train.csv")
    args = parser.parse_args()
    df = pd.read_csv(args.path).drop(columns=['ID'])
    df['Diagnosis'] = df['Diagnosis'] == 'M'

    # normalized data
    x_train, x_test, y_train, y_test = test_train_split(df, 0.8)

    x_train=(x_train-x_train.mean())/x_train.std()
    x_train = x_train.to_numpy(dtype=np.float64)
    y_train = y_train.to_numpy(dtype=np.float64)
    train(x_train, y_train)

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def train(x_train, y_train):
    epochs = 1_00
    x_len = x_train.shape[1]
    losses = []
    layers = [
        Layer(x_len, 20, layer_type="input_layer", activation_function=sigmoid),
        Layer(20, 10, layer_type="hidden_layer", activation_function=sigmoid),
        # Layer(10, 1, layer_type="output_layer", activation_function=softmax)
        Layer(10, 1, layer_type="output_layer")
    ]

    mp = MultilayerPerceptron(layers)
    for epoch in range(1, epochs):
        for batch in iterate_minibatches(x_train, y_train, 2**8, shuffle=True):
            x_batch, y_batch = batch
            p = mp.backward_propagation(x_batch.T, y_batch, epoch)
            loss = mp.loss(p, np.array(y_batch))
            losses.append(loss)
            print(f"epoch {epoch}/{epochs} - loss: {loss} - val_loss: {loss}")

    y = y_train
    x = x_train[i][np.newaxis,:].T
    i = np.random.randint(0, x_train.shape[0])
    p = mp.backward_propagation(x, y[i], epoch)
    print(f"{i}={p}, should be: {y[i]}")
    pd.DataFrame(losses).plot()
    plt.savefig('loss.png')

if __name__ == '__main__':
    main()

