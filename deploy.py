#conda info --envs got tf1 without tensorflow, tf1.0, tf2, tf2.0 without tensorflow, tf2.6
#jacaranda model has been trained with tf2.6, import cv2 after tensorflow
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input 
from tensorflow.keras.preprocessing.image import img_to_array    
import cv2

st.title('Jacaranda Identification')
st.markdown('A Deep learning application to identify if a satellite image clip contains Jacaranda trees. The predicting result will be "Jacaranda", or "Others"')

uploaded_file = st.file_uploader("Upload an image file", type="jpg")

img_height = 224
img_width = 224
class_names = ['Jacaranda', 'Others']

model = tf.keras.models.load_model('model')

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img)

    #img = cv2.resize(img, (img_height, img_width))
    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array, axis = 0) # Create a batch
    processed_image = preprocess_input(img_array)

    Generate_pred = st.button("Generate Prediction")
    if Generate_pred:
        predictions = model.predict(processed_image)
        score = predictions[0]
        
        st.markdown("Predicted class of the image is : {}".format(class_names[np.argmax(score)]))
