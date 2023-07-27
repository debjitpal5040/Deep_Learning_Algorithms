import keras
# import matplotlib.pyplot as plt
import numpy as np
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train / 255
X_test = X_test / 255
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(500, activation='sigmoid'),
    keras.layers.Dense(500, activation='sigmoid'),
    keras.layers.Dense(10, activation='sigmoid')

])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)
classes = model.predict(X_test)
print(classes)