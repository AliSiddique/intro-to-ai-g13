import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score

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
spotify_df = spotify_df[selected_features]

# Normalise the data using Min Max scaling
scaler = MinMaxScaler()
spotify_df[selected_features[:-1]] = scaler.fit_transform(spotify_df[selected_features[:-1]])

# Split data into training and testing sets
X = spotify_df[selected_features[:-1]]
y = spotify_df['danceability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set threshold
y_train_binary = np.where(y_train > threshold, 1, 0)
y_test_binary = np.where(y_test > threshold, 1, 0)

# Define the model architecture
model = Sequential([
      Dense(128, activation='leakyrelu', input_shape=(X_train.shape[1],)),
      Dense(64, activation='leakyrelu'),
      Dense(32, activation='leakyrelu'),
      Dense(16, activation='leakyrelu'),
      Dense(8, activation='leakyrelu'),
      Dense(1, activation='sigmoid')  # Sigmoid activation for binary classification
  ])

# Define the optimizer (RMSprop)
rmsprop_optimizer = tf.keras.optimizers.RMSprop()

# Compile the model
model.compile(optimizer=rmsprop_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_binary, epochs=100, batch_size=8, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test_binary, verbose=2)

# Print out the model accuracy
loss, accuracy = model.evaluate(X_test, y_test_binary)
print(f"Accuracy on test set: {accuracy}")

# Predict probabilities on test set
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > threshold).astype('int')


# Confusion Matrix
conf_matrix = confusion_matrix(y_test_binary, y_pred)
print(f"Confusion Matrix:\n{conf_matrix}")

# Classification Report
class_report = classification_report(y_test_binary, y_pred)
print(f"Classification Report:\n{class_report}")

# F1-score
f1 = f1_score(y_test_binary, y_pred)
print(f"F1-score: {f1}")

# Recall
recall = recall_score(y_test_binary, y_pred)
print(f"Recall: {recall}")