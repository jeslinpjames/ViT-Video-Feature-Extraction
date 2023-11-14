import numpy as np
import pandas as pd
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

# Load the data from the CSV files in the folder
folder_path = "D:/git/output_folder"
csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

data = []
labels = []

max_rows = 590

for csv_file in csv_files:
    df = pd.read_csv(os.path.join(folder_path, csv_file))
    df = shuffle(df)  # Shuffle the rows
    df = df.head(max_rows)  
    data.append(df.to_numpy())
    labels.extend([csv_file.split(".")[0]] * len(df))

# Extract the features and labels for each class
X = np.vstack(data)
label_encoder = LabelEncoder()
y = to_categorical(label_encoder.fit_transform(labels))  # One-hot encode labels

# Combine the data and labels into a single dataset
X_combined, y_combined = shuffle(X, y, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

# Reshape the training and testing data to match the input shape of the LSTM model
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(50, return_sequences=True, activation='relu'))
model.add(LSTM(20, return_sequences=True, activation='relu'))
model.add(LSTM(10, activation='relu'))
model.add(Dense(9, activation='softmax'))

# Compile the model with the appropriate loss function, optimizer, and metrics
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model on the training set
model.fit(X_train, y_train, epochs=25, batch_size=16, verbose=2)

# Evaluate the model on the testing set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the model weights to a file
model.save_weights("model_weights.h5")
