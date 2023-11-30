import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import os

flight_number = 1002
data = pd.read_pickle(f'Data/Dataframe_{flight_number}.pkl')

input_features = ['flux_c_t','flux_c_z','cur_ac_lo','ins_alt',
                  'vol_back_n','vol_back_p','vol_acc_n','vol_acc_p',
                  'ins_lat', 'ins_roll',
                  'mag_3_c', 'mag_4_c', 'mag_5_c',
                  'utm_x','utm_y','utm_z'] # x,y,z are not input features

output_labels = ['slg']


input_dim = len(input_features)
output_dim = len(output_labels)
sequence_length = 5
epochs = 20

path = f'Results/pre_slg_in={input_dim-3}_out={output_dim}_seq={sequence_length}_epoch={epochs}'
if not os.path.exists(path):
    os.mkdir(path)

data_out_df = data.loc[:, output_labels]
data_out = data_out_df.to_numpy(dtype = 'float32')

data_in_df = data.loc[:, input_features]
data_in = data_in_df.to_numpy(dtype = 'float32')

# Function to transform the time series data into a sequential format

def create_sequences(data, sequence_length):
    x = []

    for i in range(len(data) - sequence_length):
        x.append(data[i:i + sequence_length,:])

    x = np.array(x)

    return x

# Create the sequential data

X = create_sequences(data_in, sequence_length)
y = create_sequences(data_out, sequence_length)


# Split data into training, testing, and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=(0))
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=(0))

pos = X[:,:,-3:]
pos_train = X_train[:,:,-3:]
pos_test = X_test[:,:,-3:]
pos_val = X_val[:,:,-3:]

X = X[:,:,:-3]
X_train = X_train[:,:,:-3]
X_test = X_test[:,:,:-3]
X_val = X_val[:,:,:-3]

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(X_train))
  
# Define model
model = tf.keras.Sequential([
    normalizer,
    tf.keras.layers.LSTM(64, input_shape=(sequence_length, input_dim), return_sequences=True),
    tf.keras.layers.Dense(64),
    tf.keras.layers.Dense(64),
    tf.keras.layers.LSTM(64, input_shape=(sequence_length, input_dim), return_sequences=True),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(output_dim)
])

# Compile model
model.compile(optimizer='adam', loss='mae', metrics=['mape'])

# Train model
history = model.fit(X_train, y_train, epochs= epochs, validation_data=(X_val, y_val))

# Plot RMSE per epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('MAE per epoch')
plt.ylabel('MAE')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.savefig(f'{path}/mae.png',dpi = 600)
plt.show()

# Predict on test set
y_pred = model.predict(X_test)

# Calculate error 

error = np.zeros((sequence_length, 1))
for i in range(sequence_length):
    error[i] = np.mean(abs(y_test[:,i,0] - y_pred[:,i,0]))


# Plot prediction vs real data
plt.plot(y_test[:1000,0,0])
plt.plot(y_pred[:1000,0,0])
plt.title('Prediction vs real data')
plt.ylabel('Value')
plt.xlabel('Time step')
plt.legend(['real', 'prediction'], loc='upper right')
plt.savefig(f'{path}/slg.png',dpi = 600)
plt.show()

model.save(f'{path}/model/trained_model')

model = tf.keras.models.load_model(f'{path}/model/trained_model')

# use the trained model to predict slg
slg_train = model.predict(X_train)
slg_test = model.predict(X_test)

# keep the first prediction
slg_train = slg_train[:,0,:]
slg_test = slg_test[:,0,:]
pos_train = pos_train[:,0,:]
pos_test = pos_test[:,0,:]

slg_train_df = pd.DataFrame(slg_train, columns = ['slg'])
slg_test_df = pd.DataFrame(slg_test, columns = ['slg'])
pos_train_df = pd.DataFrame(pos_train, columns = ['x', 'y', 'z'])
pos_test_df = pd.DataFrame(pos_test, columns = ['x', 'y', 'z'])

# Random Forest 

from sklearn import ensemble

for i in range(200):
    forest_model = ensemble.RandomForestRegressor(max_depth=(i+1))
    forest_model.fit(slg_train_df, pos_train_df)
    
    pos_pred_test = forest_model.predict(slg_test_df)
    pos_pred_train = forest_model.predict(slg_train_df)
    
    error_test = pos_pred_test - pos_test
    MAE_test = np.mean(np.abs(error_test), axis = 0)
    error_train = pos_pred_train - pos_train
    MAE_train = np.mean(np.abs(error_train), axis = 0)
    print(f'depth = {i+1}, MAE_test = {MAE_test}, MAE_train = {MAE_train}')


