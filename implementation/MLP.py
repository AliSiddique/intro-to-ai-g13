import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

# Tensorboard for experimentation analysis
# %load_ext tensorboard
# %tensorboard --logdir log


# Turn data into tensors
spotify_df = pd.read_csv("../spotify_data.csv",encoding="ISO-8859-1")
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


spotify_df.danceability.unique()


# Define the model architecture
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Sigmoid activation for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Convert labels to binary classification (0 or 1) based on threshold
threshold = 0.5
y_train_binary = np.where(y_train > threshold, 1, 0)
y_test_binary = np.where(y_test > threshold, 1, 0)

# Train the model
model.fit(X_train, y_train_binary, epochs=20, batch_size=32, validation_split=0.2)


loss, mae = model.evaluate(X_test, y_test)
print(f"Mean Absolute Error (MAE) on test set: {mae}")


from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score

# Evaluate the model on test set
loss, accuracy = model.evaluate(X_test, y_test_binary)
print(f"Accuracy on test set: {accuracy}")


# Predict probabilities on test set
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > threshold).astype('int')  # Applying threshold manually

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


# Data for plotting
metrics = ['Accuracy', 'F1-score', 'Recall']
scores = [accuracy, f1, recall]

# Bar plot for performance metri
plt.figure(figsize=(8, 6))
plt.bar(metrics, scores, color=['blue', 'green', 'orange'])
plt.title('Model Performance Metrics')
plt.ylabel('Score')
plt.ylim(0, 1)  # Set y-axis limit between 0 and 1 (for accuracy and F1 score)
plt.grid(axis='y')

# Show the scores on top of the bars
for i, score in enumerate(scores):
    plt.text(i, score + 0.02, f'{score:.2f}', ha='center', color='black', fontsize=12)





# Plot confusion matrix
plt.figure(figsize=(6, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

# Add labels and ticks
classes = ['Not Dancable', 'Dancable']
tick_marks = np.arange(len(classes))

plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

# Display numerical values in cells
thresh = conf_matrix.max() / 2
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()