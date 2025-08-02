import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tkinter import Tk, Label
from PIL import Image, ImageTk

# Load the trained model
model = load_model('traffic_signal_classifier_augmented.keras')

# Function to load and preprocess images
def load_image(image_path, img_size=(224, 224)):
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0  # Normalize image
    return np.expand_dims(img_array, axis=0)

# Function to predict and display image
def display_image(image_path, label, root, img_label, result_label):
    img_array = load_image(image_path)
    prediction = model.predict(img_array)
    predicted_class = 'Emergency' if prediction[0][0] > 0.5 else 'Non-Emergency'

    # Update the prediction text in the result_label
    result_label.config(text=f'Prediction: {predicted_class}')

    # Display image
    img = load_img(image_path, target_size=(224, 224))
    img = img.resize((300, 300))  # Resize image for display in Tkinter window
    img = np.array(img)

    # Show image
    img_tk = ImageTk.PhotoImage(image=Image.fromarray(img))
    img_label.config(image=img_tk)
    img_label.image = img_tk  # Keep a reference to avoid garbage collection

# Create a tkinter window
root = Tk()
root.title("Traffic Signal Classification")

# Label to display the image
img_label = Label(root)
img_label.pack()

# Label to display the classification result
result_label = Label(root, text="Prediction: ", font=("Arial", 16))
result_label.pack()

# Function to show the next image in the folder
def show_next_image(test_images, index=0):
    if index >= len(test_images):
        index = 0  # Restart slideshow if we reach the end

    img_name = test_images[index]
    img_path = os.path.join("C:/VEHICLE/Emergency_Vehicles/test", img_name)
    
    display_image(img_path, img_name, root, img_label, result_label)
    
    # Call next image after a delay
    root.after(2000, show_next_image, test_images, index + 1)

# Load test images from the CSV
test_data = pd.read_csv("C:/VEHICLE/Emergency_Vehicles/test.csv")
test_images = test_data['image_names'].values

# Start showing images from the test folder
show_next_image(test_images, 0)

# Start Tkinter GUI loop
root.mainloop()
