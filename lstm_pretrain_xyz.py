import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np 
import h5py

filename_afterTL = 'C:/Users/Amin/iCloudDrive/1. PhD/Chaos Lab/Projects/Transfer Learning MagNav/Data/data_1002.h5'
filename_beforeTL = 'C:/Users/Amin/iCloudDrive/1. PhD/Chaos Lab/Projects/Transfer Learning MagNav/Data/Flt1002_train.h5'

sequence_length = 10
input_dim = 7
output_dim = 3

# Read 

with h5py.File(filename_afterTL, "r") as f:
    # print("Keys: %s" % f.keys())
    dset = f['tt']
    # dset.shape 
    data = np.zeros((dset.shape[0], 12))
    data[:,0] = np.array(f.get('tt'))
    data[:,1] = np.array(f.get('mag_3_c'))
    data[:,2] = np.array(f.get('mag_4_c'))
    data[:,3] = np.array(f.get('mag_5_c'))
    data[:,8] = np.array(f.get('slg'))

with h5py.File(filename_beforeTL, "r") as f:
    # print("Keys: %s" % f.keys())
    data[:,4] = np.array(f.get('flux_c_t'))
    data[:,5] = np.array(f.get('cur_ac_lo'))
    data[:,6] = np.array(f.get('ins_alt'))
    data[:,7] = np.array(f.get('flux_c_z'))
    data[:,9] = np.array(f.get('utm_x'))
    data[:,10] = np.array(f.get('utm_y'))
    data[:,11] = np.array(f.get('utm_z'))

data_in = data[:,1:8]

# x,y,z are the targets
data_out = data[:,9:]

# Normalizing Data out
scaler = MinMaxScaler()
data_out = scaler.fit_transform(data_out)

# Denormalizing 
unnormalized_data_out = scaler.inverse_transform(data_out) 

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(X_train))

# print(normalizer.mean.numpy())
# first = np.array(X_train[:1])

# with np.printoptions(precision=2, suppress=True):
#   print('First example:', first)
#   print()
#   print('Normalized:', normalizer(first).numpy())
  
# Define model
model = tf.keras.Sequential([
    normalizer,
    tf.keras.layers.LSTM(16, input_shape=(sequence_length, input_dim), return_sequences=True),
    tf.keras.layers.Dense(32),
    tf.keras.layers.Dense(32),
    tf.keras.layers.LSTM(16, input_shape=(sequence_length, input_dim), return_sequences=True),
    tf.keras.layers.Dropout(0.1),
    # tf.keras.layers.Normalization(),
    tf.keras.layers.Dense(output_dim)
])

# Compile model
model.compile(optimizer='adam', loss='mae', metrics=['mape'])

# Train model
history = model.fit(X_train, y_train, epochs= 20, validation_data=(X_val, y_val))

# Plot RMSE per epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('MAE per epoch')
plt.ylabel('MAE')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

# Predict on test set
y_pred = model.predict(X_test)


# error_0 = np.mean(abs(y_test[:,0,0] - y_pred[:,0,0]))
# error_1 = np.mean(abs(y_test[:,1,0] - y_pred[:,1,0]))
# error_2 = np.mean(abs(y_test[:,2,0] - y_pred[:,2,0]))
# error_3 = np.mean(abs(y_test[:,3,0] - y_pred[:,3,0]))
# error_4 = np.mean(abs(y_test[:,4,0] - y_pred[:,4,0]))
# error_5 = np.mean(abs(y_test[:,5,0] - y_pred[:,5,0]))
# error_6 = np.mean(abs(y_test[:,6,0] - y_pred[:,6,0]))
# error_7 = np.mean(abs(y_test[:,7,0] - y_pred[:,7,0]))
# error_8 = np.mean(abs(y_test[:,8,0] - y_pred[:,8,0]))
# error_9 = np.mean(abs(y_test[:,9,0] - y_pred[:,9,0]))


unnormalized_y_test = scaler.inverse_transform(y_test[:100,0,:]) 
unnormalized_y_pred = scaler.inverse_transform(y_pred[:100,0,:]) 

x_error = np.mean(abs(unnormalized_y_test[:,0] - unnormalized_y_pred[:,0]))
y_error = np.mean(abs(unnormalized_y_test[:,1] - unnormalized_y_pred[:,1]))
z_error = np.mean(abs(unnormalized_y_test[:,2] - unnormalized_y_pred[:,2]))

# Plot prediction vs real data
plt.plot(unnormalized_y_test[:,0])
plt.plot(unnormalized_y_pred[:,0])
plt.title('Prediction vs real data')
plt.ylabel('Value')
plt.xlabel('Time step')
plt.legend(['real', 'prediction'], loc='upper right')
plt.show()

plt.plot(unnormalized_y_test[:,1])
plt.plot(unnormalized_y_pred[:,1])
plt.title('Prediction vs real data')
plt.ylabel('Value')
plt.xlabel('Time step')
plt.legend(['real', 'prediction'], loc='upper right')
plt.show()

plt.plot(unnormalized_y_test[:,2])
plt.plot(unnormalized_y_pred[:,2])
plt.title('Prediction vs real data')
plt.ylabel('Value')
plt.xlabel('Time step')
plt.legend(['real', 'prediction'], loc='upper right')
plt.show()

# plt.plot(data_in)
# plt.plot(data_out)
# plt.title('Input Output Data')
# plt.ylabel('Value')
# plt.xlabel('Time step')
# plt.legend(['in1', 'in2', 'in3','target'], loc='upper right')
# plt.show()

model.save('saved_model/flt_1002_xyz')


