import pickle

import pandas as pd
from matplotlib import pyplot as plt, pyplot
import numpy as np
import warnings
import tensorflow as tf


warnings.filterwarnings('ignore')


if __name__ == '__main__':
    print(tf.__version__)
    # model2_ML = pd.read_csv("D:/NSBM/NSBM/year 4/research/heart disease prediction/data/Model1 Evaluation.csv")
    # print(model2_ML)
    #
    # height = model2_ML['Accuracy of Model']
    # bars = model2_ML['Model']
    # x_pos = np.arange(len(bars))
    #
    # # Create bars with different colors
    # plt.bar(x_pos, height, color=['red', 'blue', 'green', 'orange', 'yellow', 'pink', 'purple', 'brown', 'cyan'])
    # plt.xticks(x_pos, bars)
    # plt.show()

    model2_DL = pd.read_csv("D:/NSBM/NSBM/year 4/research/heart disease prediction/data/Model2 Evaluation 1.csv")
    print(model2_DL)

    height = model2_DL['Accuracy']
    bars = model2_DL['Model']
    x_pos = np.arange(len(bars))

    # Create bars with different colors
    plt.bar(x_pos, height, color=['yellow', 'blue', 'green', 'purple', 'red'])
    plt.xticks(x_pos, bars)
    plt.show()



