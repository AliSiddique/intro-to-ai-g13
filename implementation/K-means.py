import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import  f1_score, recall_score, accuracy_score, confusion_matrix
from sklearn.cluster import KMeans

# Read in the data
df = pd.read_csv('../spotify_data.csv')
df.drop(columns=['Unnamed: 0','mode','key'], inplace=True)

# Set 'track_id' as the new index
df.set_index('track_id', inplace=True)

# Set threshold
threshold = 0.5

# Apply lambda function to update danceability values
df['danceability'] = df['danceability'].apply(lambda x: 1 if x >= threshold else 0)
# Make a new csv file with the change data
df.to_csv("../updated_spotify_data.csv", index=False)

# Turn data into tensors
spotify_df = pd.read_csv("../updated_spotify_data.csv")
# Selected relevant features
selected_features = ['energy', 'loudness', 'liveness', 'instrumentalness', 'danceability']

# Select the features
spotify_df = spotify_df[selected_features]

# Normalise the data using Min Max scaling
scaler = MinMaxScaler()
spotify_df[selected_features[:-1]] = scaler.fit_transform(spotify_df[selected_features[:-1]])

# Split data into training and testing sets
X = spotify_df[selected_features[:-1]]
y = spotify_df['danceability']
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set threshold
y_train_binary = np.where(y_train > threshold, 1, 0)
y_test_binary = np.where(y_test > threshold, 1, 0)


# Initialise K means clustering algorithm
knn_classifier = KMeans(n_clusters=2)

# Train the model
knn_classifier.fit(X_train, y_train_binary)

# Predict on test set
y_pred_km = knn_classifier.predict(X_test)

# Evaluate model
accuracy_knn = accuracy_score(y_test_binary, y_pred_km)
f1_knn = f1_score(y_test_binary, y_pred_km)
recall_knn = recall_score(y_test_binary, y_pred_km)

# Print metrics
print(f"Accuracy (K-Nearest Neighbors): {accuracy_knn}")
print(f"F1-score (K-Nearest Neighbors): {f1_knn}")
print(f"Recall (K-Nearest Neighbors): {recall_knn}")

# Print confusion matrix
conf_matrix_knn = confusion_matrix(y_test_binary, y_pred_km)
print(f"Confusion Matrix (K-Nearest Neighbors):\n{conf_matrix_knn}")