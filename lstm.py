import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Load the features from CSV files
data1= "D:/git/output_folder/Adbutham.csv"
data2= "D:/git/output_folder/Bhayanakam.csv"

bayanakam_data = pd.read_csv(data1).to_numpy()
athbutham_data = pd.read_csv(data2).to_numpy()

# Assign labels for Bayanakam and Athbutham classes
bayanakam_labels = np.zeros(bayanakam_data.shape[0])
athbutham_labels = np.ones(athbutham_data.shape[0])

# Combine the data and labels
X = np.vstack((bayanakam_data, athbutham_data))
y = np.concatenate((bayanakam_labels, athbutham_labels))

# Shuffle the data
X, y = shuffle(X, y, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Assuming 'X_train' is your data
print("Shape of X_train:", X_train.shape)

# Assuming 'X_test' is your test data
print("Shape of X_test:", X_test.shape)

# Assuming 'y_train' is your labels for training data
print("Shape of y_train:", y_train.shape)

# Assuming 'y_test' is your labels for test data
print("Shape of y_test:", y_test.shape)
# ... (previous code)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])  # Reshape training data
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])  # Reshape test data

# Define the LSTM model
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(50, return_sequences=True, activation='relu'))
model.add(LSTM(20, return_sequences=True, activation='relu'))
model.add(LSTM(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=25, batch_size=16, verbose=2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save model weights to a file
model.save_weights("model_weights.h5")


model.load_weights("model_weights.h5")

# Function to predict the class of a given index
def predict_class_for_index(csv_file, index):
    # Load the CSV file and extract the features for the given index
    data = pd.read_csv(csv_file).to_numpy()
    input_data = data[index].reshape(1, 1, data.shape[1])
    
    # Make the prediction
    prediction = model.predict(input_data)
    
    # Check the class and print it
    predicted_class = 'Bayanakam' if prediction < 0.5 else 'Athbutham'
    print(f"Predicted Class for Index {index}: {predicted_class}")


flag=0
while(flag==0):
    # Example: Predict class for a specific index (change 'index' as needed)
    index_to_predict = int(input("Enter index : "))  # Change this to the index you want to predict
    file=int(input("Enter 0 for Adbutham and 1 for Bhayanakam : "))
    if(file==0):
         predict_class_for_index(data1,index_to_predict)
    else:
        predict_class_for_index(data2,index_to_predict)
    flag=int(input("Enter 0 to continue and 1 to exit : "))