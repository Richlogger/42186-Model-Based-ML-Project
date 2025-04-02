# Loading data from tsv files
import importlib_resources
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from scipy.io import loadmat
from scipy import stats

import sklearn.linear_model as lm
from sklearn import model_selection
from dtuimldmtools import rlr_validate, confmatplot, rocplot, draw_neural_net, train_neural_net
from sklearn.model_selection import train_test_split
from matplotlib.pylab import (
    figure,
    hist,
    grid,
    legend,
    loglog,
    semilogx,
    show,
    plot,
    subplot,
    title,
    xlabel,
    ylabel,
)
import matplotlib.pyplot as plt


# Load the tsv data using the Pandas library
script_directory = Path(__file__).parent
filename = script_directory / 'example'


# Print the location of the dataset file on your computer. 
print("\nLocation of the example file: {}".format(filename))

# Load the dataset file using pandas
df = pd.read_tsv(filename)

# Pandas returns a dataframe, (df) which could be used for handling the data.
# We will however convert the dataframe to numpy arrays.
raw_data = df.values






if __name__ == "__main__":
    #means_and_stds()
    #neural_net(X, attributeNames)
    pass


