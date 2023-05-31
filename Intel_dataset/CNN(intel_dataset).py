import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd

dataset_dir = 'intel_img-classification'
image_size = (150, 150)
batch_size = 32

# Create separate ImageDataGenerator instances for each dataset
train_datagen = ImageDataGenerator(rescale=1./255)  # Normalizing pixel values
test_datagen = ImageDataGenerator(rescale=1./255)

class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

train_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_dir, 'Training'),
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(dataset_dir, 'Testing'),
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
)

image_paths = [os.path.join(dataset_dir, 'Validation', filename) for filename in os.listdir(os.path.join(dataset_dir, 'Validation'))]

df = pd.DataFrame({'filename': image_paths})

val_generator = test_datagen.flow_from_dataframe(
    dataframe=df,
    x_col='filename',
    y_col=None,
    target_size=image_size,
    batch_size=batch_size,
    class_mode=None,
    # shuffle=False
)

model = keras.Sequential([
    keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    # keras.layers.Conv2D(32, (3, 3), activation='relu'),
    # keras.layers.MaxPooling2D((2, 2)),
    # keras.layers.Conv2D(16, (3, 3), activation='relu'),
    # keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='sigmoid'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(16, activation='relu'),  
    keras.layers.Dense(6, activation='softmax')
])

model = model.load(resne)

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

model.fit(train_generator, epochs=5, validation_data=test_generator)

loss, accuracy = model.evaluate(test_generator)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

model.save('intel_improved_v1.h5')

# image_paths = [os.path.join(dataset_dir, 'Validation', filename) for filename in os.listdir(os.path.join(dataset_dir, 'Validation'))]

# df = pd.DataFrame({'filename': image_paths})

# val_generator = test_datagen.flow_from_dataframe(
#     dataframe=df,
#     x_col='filename',
#     y_col=None,
#     target_size=image_size,
#     batch_size=batch_size,
#     class_mode=None,
#     shuffle=False
# )

# # Load the saved model
# model = keras.models.load_model("intel.h5")

# # Get the filenames of the images
# filenames = val_generator.filenames

# # Predict the images
# predictions = model.predict(val_generator)

# for i, filename in enumerate(filenames):
#     print(filename, class_names[predictions[i]])
