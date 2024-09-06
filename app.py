import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load the pre-trained TensorFlow model\
model_path = "FFC-Dog-Cat-Classifier/dog_cat_classifier_model.h5"

model = tf.keras.models.load_model(model_path)

# Define a function to preprocess the uploaded image
def preprocess_image(image):
    # Resize the image to the target input size of the model (e.g., 128x128)
    size = (150, 150)  # Change this based on your model input size
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Convert image to numpy array
    image = np.asarray(image)

    if image.shape[-1] == 4:  # RGBA image
        
        image = image[..., :3]
    # Normalize the image (if necessary, depends on how your model was trained)

     # Check the image shape
    if image.shape != (150, 150, 3):
        raise ValueError(f"Unexpected image shape: {image.shape}")

    # Reshape the image for the model (batch_size, height, width, channels)
    image = image.reshape((1, 150, 150, 3))  # Model expects 4D tensor (batch_size, height, width, channels)
    
    
    return image

# Define a function to make predictions
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

# Streamlit app
st.title("Image Classifier")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Make predictions when user uploads an image
    st.write("Classifying...")
    prediction = predict(image)
    
    # Display the prediction
    if prediction >= 0.5:
        st.write("It's a Dog!")
    else:
        st.write("It's a Cat!")
