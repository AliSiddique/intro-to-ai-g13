import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('../spotify_data.csv')
df.drop(columns=['Unnamed: 0','mode','key'], inplace=True)

# Set 'track_id' as the new index
df.set_index('track_id', inplace=True)


null_values = df.isnull()
