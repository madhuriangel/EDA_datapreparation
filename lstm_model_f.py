import numpy as np
import xarray as xr
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import optimizers, Sequential, Model
import tensorflow.keras.layers as L
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


file_path = 'Data_noaa_copernicus/noaa_avhrr/noaa_icesmi_combinefile.nc'
ds = xr.open_dataset(file_path)

time = ds['time'].values
lat = ds['lat'].values
lon = ds['lon'].values

# Preprocess data, handling NaN values by avoiding them
sst_data = ds['sst'].values

# Find indices of non-NaN values
non_nan_indices = np.where(~np.isnan(sst_data))

# Extract non-NaN values
sst_data = sst_data[non_nan_indices[0], non_nan_indices[1], non_nan_indices[2]]

# Reshape the data to match the LSTM input shape
sst_data = sst_data.reshape(sst_data.shape[0], -1)

# Normalize the data
scaler = MinMaxScaler()
sst_data_normalized = scaler.fit_transform(sst_data)

# Create sequences for time series prediction
sequence_length = 10
X, Y = [], []
for i in range(len(sst_data_normalized) - sequence_length):
    X.append(sst_data_normalized[i:i+sequence_length])
    Y.append(sst_data_normalized[i+sequence_length])

X, Y = np.array(X), np.array(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Build the LSTM model
lstm_model = Sequential()
lstm_model = Sequential()
lstm_model.add(L.InputLayer(shape=(X_train.shape[1], X_train.shape[2])))
lstm_model.add(L.LSTM(10, return_sequences=True))
lstm_model.add(L.LSTM(6, activation='relu',return_sequences=True))
lstm_model.add(L.LSTM(1, activation='relu'))
lstm_model.add(L.Dense(10,  activation='relu'))
lstm_model.add(L.Dense(10,  activation='relu'))
lstm_model.add(L.Dense(1))
lstm_model.summary()

adam = optimizers.Adam(learning_rate=0.001)
lstm_model.compile(loss='mse', optimizer=adam)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

lstm_history = lstm_model.fit(X_train, Y_train, 
                              epochs=20, batch_size=32, validation_split=0.2, 
                              verbose=2, callbacks=[early_stopping])
#plot
plt.figure(figsize=(12, 6))
plt.plot(lstm_history.history['loss'], label='Training Loss')
plt.plot(lstm_history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

loss = lstm_model.evaluate(X_test, Y_test)
print(f'Test Loss: {loss}')

predictions = lstm_model.predict(X_test)

# Inverse transform the predictions to the original scale
predictions = scaler.inverse_transform(predictions)


