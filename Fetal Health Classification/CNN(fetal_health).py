import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv(r"Fetal Health Classification\fetal_health.csv")

print(len(data))
predict = "fetal_health"

X = data.drop([predict], axis=1)
y = data[predict]

# Data preprocessing
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

model = keras.Sequential([
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(4, activation="softmax"),
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Convert labels to one-hot encoded vectors
y_train_one_hot = keras.utils.to_categorical(y_train) # converts integers to vecotrs of 0s and 1s eg. 1.0 = [0,1,0,0] and 3.0 = [0,0,0,1] . . .
y_test_one_hot = keras.utils.to_categorical(y_test)

model.fit(X_train, y_train_one_hot, epochs=50)

model.save('fetal_health.h5')

# Evaluate on test data
loss, accuracy = model.evaluate(X_test, y_test_one_hot)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)


