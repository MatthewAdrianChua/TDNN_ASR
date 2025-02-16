import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, models
from jiwer import wer

# Load the JSON file
with open("data.json", "r") as fp:
    data = json.load(fp)

# Extract data
mappings = data["mappings"]
labels = data["labels"]
mfccs = data["MFCCs"]
filenames = data["files"]

# Convert lists to NumPy arrays
X = np.array(mfccs)  # MFCC features
y = np.array(labels)  # Labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the TDNN model
def build_tdnn(input_shape, num_classes):
    model = models.Sequential()

    # Input layer
    model.add(layers.Input(shape=input_shape))

    # Time Delay Layers (1D Convolutions)
    model.add(layers.Conv1D(filters=64, kernel_size=5, strides=1, padding="same", activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(keras.layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2, strides=2))

    model.add(layers.Conv1D(filters=128, kernel_size=5, strides=1, padding="same", activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(keras.layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2, strides=2))

    model.add(layers.Conv1D(filters=256, kernel_size=5, strides=1, padding="same", activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(keras.layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2, strides=2))

    # Flatten the output
    model.add(layers.Flatten())

    # Fully connected layers
    model.add(layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(layers.Dropout(0.5))

    # Output layer
    model.add(layers.Dense(num_classes, activation="softmax"))

    return model

# Define input shape and number of classes
input_shape = (X_train.shape[1], X_train.shape[2])  # (time_steps, num_mfcc_features)
num_classes = len(mappings)  # Number of unique labels

# Build the model
model = build_tdnn(input_shape, num_classes)

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Print model summary
model.summary()

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=40,
    batch_size=32
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Generate predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels

# Map predictions and ground truth to words
y_test_words = [mappings[label] for label in y_test]
y_pred_words = [mappings[label] for label in y_pred_classes]

# Calculate Word Error Rate (WER)
wer_score = wer(y_test_words, y_pred_words)
print(f"Word Error Rate (WER): {wer_score:.4f}")