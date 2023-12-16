import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('../spotify_data.csv')
df

df.head()



df.drop(['Unnamed: 0'], axis=1, inplace=True)