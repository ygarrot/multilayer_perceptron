# import argparse
# import random
# import sys
import pandas as pd

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

concat = ['ID', 'Diagnosis'] + \
    [' '.join([cat, stat]) for cat in categories for stat in stats ]
df = pd.read_csv("./resources/data.csv")
df.columns = concat
print(df.head())
df.reset_index(drop=True, inplace=True)
df.to_csv("./resources/data_p.csv", index=False)

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("path", type=str, default="./resources/data.csv", help="dataset_train.csv")
#     args = parser.parse_args()

# if __name__ == '__main__':
#     main()
