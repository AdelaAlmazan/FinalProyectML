import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import joblib
from collections import Counter

# Cargar modelos previamente entrenados
svm_model = joblib.load('svm_model.sav')
decision_tree_model = joblib.load('decisionTreeClassifier.sav')
logistic_model = joblib.load('logistic_model.sav')
random_forest = joblib.load('RandomForest.sav')

# Cargar el extractor de características
url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
mobilenetv2 = tf.keras.Sequential([
    hub.KerasLayer(url, input_shape=(224,224,3))
])

# Mapeo de índices a etiquetas de emociones
emotion_labels = {0: "angry", 1: "happy", 2: "relaxed", 3: "sad"}

# Función para preprocesar la imagen
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = image / 255.0
    return image.reshape(1, 224, 224, 3)

# Función para realizar predicciones
def predict(model, image):
    features = mobilenetv2.predict(image)
    return model.predict(features)

# Función para obtener la etiqueta de la emoción
def predict_and_label(model, image):
    prediction = predict(model, image)
    emotion = emotion_labels[prediction[0]]
    return emotion

# Función para realizar ensamble de predicciones
def ensemble_predictions(image):
    predictions = [
        predict_and_label(svm_model, image),
        predict_and_label(decision_tree_model, image),
        predict_and_label(logistic_model, image),
        predict_and_label(random_forest, image)
    ]
    # Realizar una votación mayoritaria
    most_common = Counter(predictions).most_common(1)
    return most_common[0][0]

# Crear la interfaz de Streamlit
st.title("Clasificador de Emociones de Perros")

# Cargar archivo de imagen
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen cargada.', use_column_width=True)
    st.write("")
    st.write("Clasificando...")

    # Preprocesar y predecir
    preprocessed_image = preprocess_image(image)
    emotion_svm = predict_and_label(svm_model, preprocessed_image)
    emotion_dt = predict_and_label(decision_tree_model, preprocessed_image)
    emotion_lr = predict_and_label(logistic_model, preprocessed_image)
    emotion_rf = predict_and_label(random_forest, preprocessed_image)
    
    # Ensamble de predicciones
    ensemble_prediction = ensemble_predictions(preprocessed_image)

    # Mostrar los resultados
    st.write(f"Predicción con SVM: {emotion_svm}")
    st.write(f"Predicción con Decision Tree: {emotion_dt}")
    st.write(f"Predicción con Logistic Regression: {emotion_lr}")
    st.write(f"Predicción con Random Forest: {emotion_rf}")
    st.write(f"Predicción del Ensamble: {ensemble_prediction}")
