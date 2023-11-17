import numpy as np
import pandas as pd
import os
import tensorflow as tf
from image_feature_extraction_with_ViT import extract_features_from_frame
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import pickle
import random
import matplotlib.pyplot as plt
import seaborn as sns


# Load the data from the CSV files in the folder
folder_path = "D:/git/output_folder"
csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

data = []
labels = []

max_rows = 1003

for csv_file in csv_files:
    df = pd.read_csv(os.path.join(folder_path, csv_file))
    df = shuffle(df)  # Shuffle the rows
    df = df.head(max_rows)  
    data.append(df.to_numpy())
    labels.extend([csv_file.split(".")[0]] * len(df))

tf.random.set_seed(42)
random.seed(42)
np.random.seed(42)
# Extract the features and labels for each class
X = np.vstack(data)
label_encoder = LabelEncoder()
y = to_categorical(label_encoder.fit_transform(labels))  # One-hot encode labels
with open ("label_encoder.pkl", 'wb') as f:
    pickle.dump(label_encoder, f)

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

# Train the on the training set
history = model.fit(X_train, y_train, epochs=25, batch_size=16, verbose=2)

# Evaluate the model on the testing set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the model weights to a file

model.save("model.h5")
model.save_weights("model_weights.h5")
# weights = model.get_weights()
# for i, weight_array in enumerate(weights):
#     np.save(f"weight_{i}.npy", weight_array)

# # Load the saved weights using NumPy
# loaded_weights = [np.load(f"weight_{i}.npy") for i in range(len(weights))]

# # Set the loaded weights to the model
# model.set_weights(loaded_weights)



y_pred = np.argmax(model.predict(X_test),axis=1)

yy_test = np.argmax(y_test,axis=1)
print(yy_test)
print(y_test)
print(y_pred)

from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_pred,yy_test)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


cr = classification_report(y_pred,yy_test)
print(cr)

