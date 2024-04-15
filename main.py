import streamlit as st
from PIL import Image
import numpy as np

from keras.models import load_model

# Load the trained model
model = load_model('models/version_32_32.h5')
size = model.input_shape

# Define classes
classes = {
    4: ('nv', 'Melanocytic Nevi'),
    6: ('mel', 'Melanoma'),
    2: ('bkl', 'Benign Keratosis-like Lesions'),
    1: ('bcc', 'Basal Cell Carcinoma'),
    5: ('vasc', 'Pyogenic Granulomas and Hemorrhage'),
    0: ('akiec', 'Actinic Keratoses and Intraepithelial Carcinomae'),
    3: ('df', 'Dermatofibroma')
}

# Function to process uploaded image


def process_image(image):
    image = image.resize((size[1], size[2]))
    image = np.asarray(image) / 255.0
    i_array = np.array([image])
    return i_array

# Main function


def main():
    st.title('DermaCare - Skin Lesion Classifier')

    uploaded_image = st.file_uploader(
        "Please upload an image (JPG, JPEG, or PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Classify Image'):

            # Process the uploaded image
            i_array = process_image(image)

            # Make prediction
            prediction = model.predict(i_array)
            predicted_class_index = np.argmax(prediction)
            predicted_class = classes[predicted_class_index]

            st.subheader(
                f"Predicted Class: {predicted_class[0]} - {predicted_class[1]}")


# Run the app
if __name__ == '__main__':
    main()
