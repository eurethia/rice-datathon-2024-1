import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
from keras.utils import to_categorical
import os
wandb.init(config={"bs": 64})
X = None
y = None
for file in os.scandir("./bassband_data_new"):
    if 'X' in file.name:
        if X is None:
            X = np.load(file.path)
        else:
            X = np.concatenate((X, np.load(file.path)), axis=0)
    else:
        if y is None:
            y = np.load(file.path)
        else:
            y = np.concatenate((y, np.load(file.path)), axis=0)



y = y - 1

y_onehot = to_categorical(y)

# X = X.reshape(X.shape[0], -1)
print(X.shape)
print(y_onehot.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.1, random_state=42)
np.save("X_validation_data.npy", X_test)
np.save("y_validation_data.npy", y_test)
new_X_train = []
new_X_test = []
for i in range(X_train.shape[0]):

    X_train_scaled = X_train[i][:3,:]
    X_train_scaled = np.append(X_train_scaled, X_train[i][4:5,:], axis=0)
    X_train_scaled = np.transpose(X_train_scaled)
    new_X_train.append(X_train_scaled)
for i in range(X_test.shape[0]):
    scaler = StandardScaler()
    X_test_scaled = X_test[i][:3,:]
    X_test_scaled = np.append(X_test_scaled, X_test[i][4:5,:],axis=0)
    X_test_scaled = np.transpose(X_test_scaled)
    new_X_test.append(X_test_scaled)
    # X_test_scaled = scaler.transform(X_test[i])
new_X_train = np.array(new_X_train)
new_X_test = np.array(new_X_test)
scaler = StandardScaler()
new_X_train = scaler.fit_transform(new_X_train.reshape((-1,4))).reshape(new_X_train.shape)
scaler = StandardScaler()
new_X_test = scaler.fit_transform(new_X_test.reshape((-1,4))).reshape(new_X_test.shape)
print(new_X_train.shape)
print(new_X_test.shape)
INPUT_SHAPE = new_X_train[0].shape
print(INPUT_SHAPE)
from keras.models import Sequential,Model
from keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, LSTM, BatchNormalization, Dense, Input, concatenate
import keras
input_layer = Input(shape=INPUT_SHAPE, dtype="float32")
from keras import backend as K
import tensorflow as tf
from tensorflow.keras.models import load_model

# print(tf.config.list_physical_devices('GPU'))
# print(K.backend())
conv = Sequential([
    Conv1D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=INPUT_SHAPE),
    MaxPooling1D(pool_size=3),
    Dropout(0.02),
    Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'),
    MaxPooling1D(pool_size=3),
    Dropout(0.02),
    Conv1D(filters=20, kernel_size=2, strides=1, padding='same', activation='relu'),
    MaxPooling1D(pool_size=3),
    Dropout(0.02),
    Conv1D(filters=16, kernel_size=8, strides=1, padding='same', activation='relu'),
    MaxPooling1D(pool_size=3),
    Dropout(0.02),
    Conv1D(filters=10, kernel_size=3, strides=1, padding='same', activation='relu'),
    MaxPooling1D(pool_size=3),
    Dropout(0.02),
    Flatten()
])

lstm = Sequential([
    LSTM(64, return_sequences=True, recurrent_dropout=0.02),
    BatchNormalization(),
    LSTM(20, return_sequences=True, recurrent_dropout=0.02),
    Flatten(),
])

conv_output = conv(input_layer)
lstm_output = lstm(input_layer)
concatenated = concatenate([conv_output, lstm_output])

output = Dense(6, activation='softmax')(concatenated)

model = Model(inputs=input_layer, outputs=output)
learning_rate_fn = keras.optimizers.schedules.InverseTimeDecay(0.001, 1.0, 0.01)
model.compile(optimizer=keras.optimizers.AdamW(
    learning_rate=learning_rate_fn
), loss='categorical_crossentropy', metrics=['CategoricalAccuracy',"AUC"])


print(new_X_train.shape)

history = model.fit(new_X_train, y_train,batch_size=64, epochs=10,validation_data=(new_X_test, y_test),callbacks=[
    WandbMetricsLogger(log_freq="batch"),
    WandbModelCheckpoint("models3{epoch:02d}")])

score = model.evaluate(new_X_test, y_test)