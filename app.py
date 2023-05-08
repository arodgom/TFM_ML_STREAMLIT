# Importamos las librerías necesarias
import streamlit as st
import pandas as pd
import pickle
from PIL import Image

# Cargamos el modelo de predicción
model = pickle.load(open('model.pkl', 'rb'))

# Título y solicitud de los datos
st.header("Clasificación de Iris dataset:")
image = Image.open('image.png')
st.image(image, use_column_width=True)
st.write("Por favor, ingrese los valores para obtener la predicción del tipo de Iris")

SepalLengthCm = st.slider('SepalLengthCm:', 2.0, 6.0)
SepalWidthCm = st.slider('SepalWidthCm:', 0.0, 5.0)
PetalLengthCm = st.slider('PetalLengthCm',0.0, 3.0)
PetalWidthCm = st.slider('PetalWidthCm:', 0.0, 2.0)

# Recojo los datos en un dataframe
data = {'SepalLengthCm': SepalLengthCm,
        'SepalWidthCm': SepalWidthCm,
        'PetalLengthCm': PetalLengthCm,
        'PetalWidthCm': PetalWidthCm}

features = pd.DataFrame(data, index=[0])

# Calculo y muestro la predicción en base al modelo
pred_proba = model.predict_proba(features)

st.subheader('Porcentajes de la predicción:') 
st.write('**La probabilidad de que sea de clase Iris-setosa es (en %)**:',pred_proba[0][0]*100)
st.write('**La probabilidad de que sea de clase Iris-versicolor es (en %)**:',pred_proba[0][1]*100)
st.write('**La probabilidad de que sea de clase Iris-virginica es (en %)**:',pred_proba[0][2]*100)


