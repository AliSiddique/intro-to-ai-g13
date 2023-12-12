import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Tensorboard for experimentation analysis
# %load_ext tensorboard
# %tensorboard --logdir log


# Turn data into tensors
spotify_df = pd.read_csv("../spotify_data.csv",encoding="ISO-8859-1")
spotify_df.head()