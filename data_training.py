import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical

from keras.layers import Input, Dense
from keras.models import Model

is_init = False
size = -1
label = []
dictionary = {}
c = 0

for i in os.listdir():
    if i.endswith(".npy"):
        data = np.load(i)
        if not(is_init):
            is_init = True
            X = data
            size = X.shape[0]
            Y = np.array([i.split(".")[0]]*size).reshape(-1,1)
        else:
            if data.shape[1:] !=X.shape[1:]:
                raise ValueError(f"Shape mismatch: expected {X.shape[1:]}, got {data.shape[1:]}")
            X = np.concatenate((X, data))
            Y = np.concatenate((Y, np.array([i.split(".")[0]]*size).reshape(-1,1)))
        
        label.append(i.split(".")[0])
        dictionary[i.split(".")[0]] = c
        c += 1


for i in range(Y.shape[0]):
    Y[i, 0] = dictionary[Y[i, 0]]
Y = np.array(Y, dtype="int32")


Y = to_categorical(Y)


X_new = X.copy()
Y_new = Y.copy()
counter = 0

cnt = np.arange(X.shape[0])
np.random.shuffle(cnt)

for i in cnt:
    X_new[counter] = X[i]
    Y_new[counter] = Y[i]
    counter += 1

print(f"Input shape: {X.shape}")
print(f"Output shape: {Y.shape}")

ip = Input(shape=(X.shape[1],))
m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)
op = Dense(Y.shape[1], activation="softmax")(m)

model = Model(inputs=ip, outputs=op)

model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

model.fit(X, Y, epochs=50)

model.save("model.h5")
np.save("labels.npy", np.array(label))

