from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, recall_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

spotify_df = pd.read_csv('../spotify_data.csv')
selected_features = ['energy', 'loudness', 'liveness', 'instrumentalness', 'danceability']
spotify_df = spotify_df[selected_features]

# Normalise the data using Min-Max scaling
scaler = MinMaxScaler()
spotify_df[selected_features[:-1]] = scaler.fit_transform(spotify_df[selected_features[:-1]])

# Split data into training and testing sets
X = spotify_df[selected_features[:-1]]
y = spotify_df['danceability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert labels to binary classification (0 or 1) based on threshold
threshold = 0.5
y_train_binary = np.where(y_train > threshold, 1, 0)
y_test_binary = np.where(y_test > threshold, 1, 0)

# Initialize Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train_binary)

# Predict on test set
y_pred_rf = rf_classifier.predict(X_test)

# Evaluate model
accuracy_rf = accuracy_score(y_test_binary, y_pred_rf)
f1_rf = f1_score(y_test_binary, y_pred_rf)
recall_rf = recall_score(y_test_binary, y_pred_rf)

# Print metrics
print(f"Accuracy (Random Forest): {accuracy_rf}")


import time

# Start timer
start_time = time.time()

# Train the model
rf_classifier.fit(X_train, y_train_binary)

# End timer and print training time
training_time = time.time() - start_time
print(f"Training Time: {training_time:.2f} seconds")

# Start timer
start_time = time.time()

# Predict on test set
y_pred_rf = rf_classifier.predict(X_test)

# End timer and print prediction time
prediction_time = time.time() - start_time
print(f"Prediction Time: {prediction_time:.2f} seconds")



# Define different numbers of trees to test
n_estimators_list = [10, 50, 100, 200]

# Store metrics for each model
metrics = {'n_estimators': [], 'accuracy': [], 'f1_score': [], 'recall': []}

for n in n_estimators_list:
    # Initialize the Random Forest with the current number of trees
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    
    # Train the model
    rf.fit(X_train, y_train_binary)
    
    # Predict on test set
    y_pred = rf.predict(X_test)
    
    # Store metrics
    metrics['n_estimators'].append(n)
    metrics['accuracy'].append(accuracy_score(y_test_binary, y_pred))
    metrics['f1_score'].append(f1_score(y_test_binary, y_pred))
    metrics['recall'].append(recall_score(y_test_binary, y_pred))




    # Plot Accuracy vs Number of Estimators
plt.figure(figsize=(10, 6))
plt.plot(metrics['n_estimators'], metrics['accuracy'], label='Accuracy', marker='o')
plt.plot(metrics['n_estimators'], metrics['f1_score'], label='F1 Score', marker='o')
plt.plot(metrics['n_estimators'], metrics['recall'], label='Recall', marker='o')
plt.xlabel('Number of Estimators')
plt.ylabel('Scores')
plt.title('Model Performance for Different Number of Estimators')
plt.legend()
plt.grid(True)
plt.show()