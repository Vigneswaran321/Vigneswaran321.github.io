import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20, 10

from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler

# Read in the data
df = pd.read_csv("NSE-TATA.csv")

# Convert the date column to datetime
df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")

# Set the index to the date column
df.index = df['Date']

# Plot the closing price
plt.figure(figsize=(16,8))
plt.plot(df["Close"], label='Close Price history')

# Sort the data by date
data = df.sort_index(ascending=True, axis=0)

# Create a new dataframe with only the date and close columns
new_dataset = pd.DataFrame(index=range (0,len(df)),columns=['Date','Close'])

# Fill the new dataframe with the date and close values from the original dataframe
for i in range(0,len(data)):
    new_dataset["Date"][i] = data['Date'][i]
    new_dataset["Close"][i] = data['Close'][i]

# Scale the close values
scaler = MinMaxScaler(feature_range=(0,1))
final_dataset = new_dataset.values

# Split the data into train and validation sets
train_data = final_dataset[0:987,:]
valid_data = final_dataset[987:,:]

# Set the index of the new dataframe to the date column
new_dataset.index = new_dataset.Date

# Drop the date column
new_dataset.drop("Date", axis=1, inplace=True)

# Scale the close values
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(final_dataset)

# Create training and test sets
x_train_data, y_train_data = [], []
for i in range(60,len(train_data)):
    x_train_data.append(scaled_data[i-60:i,0])
    y_train_data.append(scaled_data[i,0])
x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)
x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

# Create and compile the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_data.shape[1],1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))

# Fit the model to the training data
lstm_model.compile(loss='mean_squared_error', optimizer='adam')
lstm_model.fit(x_train_data, y_train_data, epochs=1, batch_size=1, verbose=2)

# Get the validation data and scale it
inputs_data = new_dataset[len(new_dataset)-len(valid_data)-60:].values
inputs_data = inputs_data.reshape(-1,1)
inputs_data = scaler.transform(inputs_data)

# Create the test set
X_test = []
for i in range(60,inputs_data.shape[0]):
    X_test.append(inputs_data[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Use the model to predict the closing price
predicted_closing_price = lstm_model.predict(X_test)

# Inverse the scaling on the predicted closing price
predicted_closing_price = scaler.inverse_transform(predicted_closing)

# Save the model
lstm_model.save("saved_model.h5")


