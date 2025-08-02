import numpy as np
import cv2
import os
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths to your Kaggle dataset (adjust paths to your setup)
train_data_dir = 'C:\\VEHICLE\\accident\\data\\train'  # Directory containing training images (with subfolders for classes)
validation_data_dir = 'C:\\VEHICLE\\accident\\data\\train\\val'  # Directory containing validation images

# Load the pre-trained model (EfficientNetB0) as a feature extractor
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Freeze the base model (initially)
base_model.trainable = False

# Build the new model on top of the pre-trained base model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')  # Adjust the number of classes (2 for binary classification)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Data Augmentation and Normalization for training
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Data normalization for validation
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Load the data using the ImageDataGenerator
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'  # Use 'categorical' if more than 2 classes
)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'  # Use 'categorical' if more than 2 classes
)

# Correct the file path to end with '.keras'
checkpoint = ModelCheckpoint(r'C:\accident_detection_model.keras', 
                             save_best_only=True, monitor='val_loss', mode='min')


early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)

# Train the model
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator,
    callbacks=[checkpoint, early_stopping, reduce_lr]
)

# Save the final model
model.save('accident_detection_model.h5')

# Evaluate the model on validation data
loss, accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")