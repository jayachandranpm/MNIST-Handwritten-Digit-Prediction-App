import streamlit as st
import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the pre-trained model
model = load_model('hand_model.h5')

def preprocess_image(image):
    # Resize the image to 28x28 (MNIST digit size)
    resized_image = image.resize((28, 28))

    # Convert the image to a numpy array
    image_array = np.array(resized_image)

    # Normalize pixel values to be between 0 and 1
    normalized_image = image_array / 255.0

    # Flatten the image array to match the input shape of the model
    flattened_image = normalized_image.reshape(1, -1)

    return flattened_image

def make_prediction(image):
    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Make a prediction using the loaded model
    prediction = model.predict(preprocessed_image)

    # Get the predicted class label
    predicted_label = np.argmax(prediction)

    return predicted_label

def main():
    st.title("MNIST Handwritten Digit Prediction App")

    # Option to select an image from the 'test_samples' folder
    folder_path = "testSample"
    images = sorted(os.listdir(folder_path))

    # Dropdown menu to select an image
    selected_image = st.selectbox("Select an image for prediction:", images)

    if selected_image:
        image_path = os.path.join(folder_path, selected_image)
        image = Image.open(image_path)
        st.image(image, caption='Selected Image', use_column_width=True)

        # Make a prediction
        predicted_label = make_prediction(image)

        # Display the prediction result
        st.subheader("Prediction Result:")
        st.write(f"The model predicts that the digit is: {predicted_label}")

if __name__ == "__main__":
    main()
