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

def init_mp(x_len):
    layers = [
        Layer(x_len, 20, layer_type="input_layer", activation_function=sigmoid),
        Layer(20, 10, layer_type="hidden_layer", activation_function=sigmoid),
        # Layer(10, 1, layer_type="output_layer", activation_function=softmax)
        Layer(10, 1, layer_type="output_layer")
    ]
    return MultilayerPerceptron(layers)

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

    x_test=(x_test-x_test.mean())/x_test.std()
    x_test = x_test.to_numpy(dtype=np.float64)
    y_test = y_test.to_numpy(dtype=np.float64)

    mp = train(x_train, y_train, init_mp(x_train.shape[1]))
    print((mp.backward_propagation(x_test.T, y_test, 0, 0.1) > 0.5).astype(dtype=np.float64))
    print(y_test)

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

def train(x_train, y_train, mp):
    epochs = 1_00
    # epochs = 5
    losses = []
    batchsize = 2**4
    decay_rate=0.1
    lr = 0.1
    for epoch in range(1, epochs):
        # lr *= 1/1+(decay_rate*epoch)
        for batch in iterate_minibatches(x_train, y_train, batchsize, shuffle=True):
            x_batch, y_batch = batch
            p = mp.backward_propagation(x_batch.T, y_batch, lr, epoch)
        loss = mp.loss(p, np.array(y_batch))
        losses.append(loss)
        print(f"epoch {epoch}/{epochs} - loss: {loss} - val_loss: {loss}")

    pd.DataFrame(losses).plot()
    plt.savefig('loss.png')
    print(batchsize)
    return mp

if __name__ == '__main__':
    main()

