import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def train_model():
    trainFile = open('agents/agent_supervised/data_set.txt', 'r')
    csv = trainFile.readline()
    trainSet = pd.read_csv(csv, header=0)

def test_model():
    return 1