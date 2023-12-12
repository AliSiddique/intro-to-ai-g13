import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../spotify_data.csv")


print(df.columns)

print(df.describe())



numeric_data = df.select_dtypes(include=[np.number])  # Select only numeric columns

correlation_matrix = np.corrcoef(numeric_data.values, rowvar=False)
plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Correlation')
plt.title('Heatmap of Column Similarities')
plt.xticks(np.arange(len(numeric_data.columns)), numeric_data.columns, rotation=90)
plt.yticks(np.arange(len(numeric_data.columns)), numeric_data.columns)
plt.tight_layout()
plt.show()




genre_counts = df['genre'].value_counts()
plt.figure(figsize=(10, 6))
genre_counts.plot(kind='bar', color='skyblue')
plt.title('Counts of Different Genres')
plt.xlabel('Genres')
plt.ylabel('Counts')
plt.xticks(rotation=45)
plt.tight_layout() 
plt.show()



plt.figure(figsize=(8, 6))
plt.hist(df['popularity'], bins=20, color='salmon', alpha=0.7)
plt.title('Distribution of Popularity')
plt.xlabel('Popularity')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()



plt.figure(figsize=(8, 6))
plt.scatter(df['danceability'], df['energy'], color='green', alpha=0.5)
plt.title('Danceability vs Energy')
plt.xlabel('Danceability')
plt.ylabel('Energy')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()



yearly_popularity = df.groupby('year')['popularity'].mean()
plt.figure(figsize=(10, 6))
yearly_popularity.plot(kind='line', marker='o', color='blue')
plt.title('Popularity Trend Over Years')
plt.xlabel('Year')
plt.ylabel('Average Popularity')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()



plt.figure(figsize=(10, 6))
df.boxplot(column='acousticness', by='genre', vert=False, figsize=(10, 8))
plt.title('Distribution of Acousticness by Genre')
plt.xlabel('Acousticness')
plt.ylabel('Genre')
plt.tight_layout()
plt.show()



# Histogram for 'popularity'
plt.figure(figsize=(8, 6))
plt.hist(df['popularity'], bins=20, color='skyblue', alpha=0.7)
plt.title('Popularity Distribution')
plt.xlabel('Popularity')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Histogram for 'year'
plt.figure(figsize=(8, 6))
plt.hist(df['year'], bins=20, color='salmon', alpha=0.7)
plt.title('Year Distribution')
plt.xlabel('Year')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Histogram for 'danceability'
plt.figure(figsize=(8, 6))
plt.hist(df['danceability'], bins=20, color='lightgreen', alpha=0.7)
plt.title('Danceability Distribution')
plt.xlabel('Danceability')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()