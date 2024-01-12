import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier

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

# Initialise Random Forest Classifier
rf = RandomForestClassifier(n_estimators=300,max_depth=10,min_samples_leaf=8, min_samples_split=10, random_state=42)
    
# Train the model
rf.fit(X_train, y_train_binary)


# Predict on the test set
y_pred = rf.predict(X_test)
y_pred_rf = rf.predict(X_test)

# Evaluate model performance on the test set
accuracy_rf = accuracy_score(y_test_binary, y_pred_rf)
f1_rf = f1_score(y_test_binary, y_pred_rf)
recall_rf = recall_score(y_test_binary, y_pred_rf)
confusion_matrix_rf = confusion_matrix(y_test_binary, y_pred_rf)
classification_report_rf = classification_report(y_test_binary, y_pred_rf)

# Print metrics and training time
print(f"Accuracy: {accuracy_rf}")
print(f"F1 Score: {f1_rf}")
print(f"Recall: {recall_rf}")
print(f"Confusion Matrix:\n{confusion_matrix_rf}")
print(f"Classification Report:\n{classification_report_rf}")
