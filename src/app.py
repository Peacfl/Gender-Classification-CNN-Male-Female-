import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

# Load model

model_path = hf_hub_download(
    repo_id="Peacfl/Gender-Classification-CNN-Male-Female",
    filename="gender_model.keras"
)

model = tf.keras.models.load_model(model_path)

st.title("Male-Female Gender Classification")
st.write("Upload a face image and the model will classify it.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", width=250)

    # Preprocess
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]
    label = "Male" if pred > 0.5 else "Female"

    st.write(f"### Prediction: **{label}**")

    st.write(f"Confidence: `{pred:.3f}`")
