import os
import numpy as np
import pandas as pd
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# Load the dataset
data = pd.read_csv("C:/VEHICLE/Emergency_Vehicles/train.csv")

# Function to load and preprocess images
def load_images(image_names, img_size=(224, 224)):
    images = []
    for img_name in image_names:
        img = load_img(os.path.join("C:/VEHICLE/Emergency_Vehicles/train", img_name), target_size=img_size)
        img_array = img_to_array(img) / 255.0  # Normalize image to [0,1]
        images.append(img_array)
    return np.array(images)

# Load images and labels
images = load_images(data['image_names'])
labels = data['emergency_or_not'].values

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Augment minority class
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Separate minority class samples
minority_indices = np.where(y_train == 1)[0]  # Find indices of the minority class
minority_images = X_train[minority_indices]  # Extract minority class images
minority_labels = y_train[minority_indices]

# Generate augmented images
augmented_images = []
augmented_labels = []
for img in minority_images:
    img = img.reshape((1,) + img.shape)  # Reshape to 4D tensor (batch size, height, width, channels)
    for batch in datagen.flow(img, batch_size=1):  # Generate augmented images
        augmented_images.append(batch[0])  # Append the image
        augmented_labels.append(1)  # Corresponding label
        if len(augmented_images) >= len(minority_images):  # Limit augmentation
            break

# Append augmented data to the training set
X_train_augmented = np.concatenate([X_train, np.array(augmented_images)])
y_train_augmented = np.concatenate([y_train, np.array(augmented_labels)])

# Compute class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_augmented), y=y_train_augmented)
class_weights = dict(enumerate(class_weights))

# Load the pretrained MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Unfreeze top layers for fine-tuning
base_model.trainable = True
fine_tune_at = 100  # Unfreeze layers starting from this index
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Add custom top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1, activation='sigmoid')(x)

# Define the full model
model = Model(inputs=base_model.input, outputs=x)

# Compile the model with a lower learning rate
model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model with the augmented training data and class weights
history = model.fit(datagen.flow(X_train_augmented, y_train_augmented, batch_size=32),
                    validation_data=(X_val, y_val),
                    epochs=10,
                    class_weight=class_weights)

# Save the fine-tuned model
model.save('traffic_signal_classifier_augmented.keras')

# Evaluate the model on the validation data
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation accuracy: {val_accuracy}")
