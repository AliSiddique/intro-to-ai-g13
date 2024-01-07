import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('../updated_spotify_data.csv')
df.drop(columns=['Unnamed: 0','mode','key'], inplace=True)

# Set 'track_id' as the new index
df.set_index('track_id', inplace=True)


null_values = df.isnull()
null_counts = null_values.sum()
df = df.head(150000)



# Set threshold
threshold = 0.5

# Apply lambda function to update danceability values
df['danceability'] = df['danceability'].apply(lambda x: 1 if x >= threshold else 0)
# Make a new csv file with the change data
df.to_csv("../updated_spotify_data.csv", index=False)