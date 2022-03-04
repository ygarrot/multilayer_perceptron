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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, default="./resources/data.csv", help="dataset_train.csv")
    args = parser.parse_args()
    df = pd.read_csv(args.path).drop(columns=['ID'])
    df['Diagnosis'] = df['Diagnosis'] == 'M'
    # normalized data
    x_train, x_test, y_train, y_test = test_train_split(df.head(100), 0.8)

    x_train=(x_train-x_train.mean())/x_train.std()
    x_train = x_train.to_numpy(dtype=np.float64)
    y_train = y_train.to_numpy(dtype=np.float64)
    train(x_train, y_train)

def train(x_train, y_train):
    epochs = 1_000
    x_len = x_train.shape[1]
    layers = [
        Layer(x_len, x_len, layer_type="input_layer", activation_function=relu),
        Layer(x_len, x_len, layer_type="hidden_layer", activation_function=relu),
        Layer(x_len, x_len, layer_type="hidden_layer", activation_function=relu),
        Layer(x_len, 1, layer_type="output_layer")
    ]

    mp = MultilayerPerceptron(layers)
    losses = []
    for epoch in range(epochs):
        print(x_train.shape, y_train.shape)
        p = mp.backward_propagation(x_train.T, np.array(y_train).T)
        loss = mp.loss(p, y_train)
        losses.append(loss)
        # print(f"epoch {epoch}/{epochs} - loss: {loss} - val_loss: {loss}")

    pd.DataFrame(losses).plot()
    plt.show()

if __name__ == '__main__':
    main()

