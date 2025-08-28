
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import pickle

# Load the trained ANN model
#model = pickle.load(open("iris_ann_model.pkl", "rb"))
#scaler = pickle.load(open("scaler.pkl", "rb"))

#model = pickle.load(open("/content/drive/MyDrive/ICAIL_Iris_ANN/models/iris_ann_model.pkl", "rb"))
#scaler = pickle.load(open("/content/drive/MyDrive/ICAIL_Iris_ANN/models/scaler.pkl", "rb"))

model = pickle.load(open("models/iris_ann_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

# App title and intro
st.title("🌸 Iris Flower Classifier")
st.markdown("### Predict the species of an Iris flower using our trained ANN model!")

st.sidebar.header("Adjust Flower Features 🌱")
# Input sliders
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.3, 7.9, 5.8)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.4, 3.0)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 6.9, 4.3)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

# Prepare input for prediction
features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
scaled_features = scaler.transform(features)
prediction = model.predict(scaled_features)
#probabilities = model.predict_proba(scaled_features)
probabilities = model.predict(scaled_features)

species = ["Setosa 🌸", "Versicolor 🌱", "Virginica 🌺"]

# Display prediction results
st.markdown("## 🌟 Prediction Results")
st.markdown(f"**Predicted Species:** {species[np.argmax(prediction)]}")
st.markdown(f"**Confidence:** {probabilities[0][np.argmax(probabilities)] * 100:.2f}%")

# Display probability table
st.markdown("### 📊 Class Probabilities")
st.dataframe({
    "Class": species,
    "Probability (%)": [f"{p*100:.2f}%" for p in probabilities[0]]
})

st.markdown("---")
st.markdown("💡 *Tip: Adjust the sliders in the sidebar to test different flower measurements.*")


# Load the trained ANN model
# model = load_model('/content/drive/MyDrive/ICAIL_Iris_ANN/iris_ann_model.h5')

# Load the saved scaler
# scaler = joblib.load('/content/drive/MyDrive/ICAIL_Iris_ANN/models/scaler.pkl')

# App title
# st.title("🌸 Iris Flower Prediction App")

# st.write("""
# Adjust the sliders to input the flower's measurements,
# and the ANN model will predict its species.
# """)

# Create sliders for input features
# sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
# sepal_width  = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
# petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
# petal_width  = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2)

# Prepare input data and apply scaling
# input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
# input_data_scaled = scaler.transform(input_data)

# Make prediction
# prediction = model.predict(input_data_scaled)
# predicted_class = np.argmax(prediction, axis=1)[0]
# class_names = ['Setosa', 'Versicolor', 'Virginica']

# Display result
# st.subheader("Prediction")
# st.write(f"The predicted species is: **{class_names[predicted_class]}** 🌱")
# st.write(f"Confidence: **{np.max(prediction) * 100:.2f}%**")
