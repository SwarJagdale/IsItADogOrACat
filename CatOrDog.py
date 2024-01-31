import streamlit as st 
from PIL import Image 
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
st.title("Cat or Dog Classifier")
st.header("Upload a picture of a cat or dog and we will tell you which one it is if you didn't know already. ")
uploaded_file = st.file_uploader("Choose a file")
model = tf.keras.models.load_model('CatOrDog.h5')
if uploaded_file is not None:

    st.image(uploaded_file,width=700)
 

    test_image1=image.load_img(uploaded_file,
                         target_size=(64,64))
    test_image1=image.img_to_array(test_image1)
    test_image1=np.expand_dims(test_image1,axis=0)
    result1= model.predict(test_image1)
    if result1[0][0]>0.5:
        st.warning("This is a Dog.")
    
    else:
        st.warning("This is a Cat.")