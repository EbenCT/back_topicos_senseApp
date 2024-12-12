import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from data_loader import DataLoader

# Configuración
longitud, altura = 200, 200
modelo = './modelo/modelo.keras'
pesos_modelo = './modelo/pesos.weights.h5'

# Variables globales para el modelo y las clases, inicializadas como None
cnn = None
classes = None

def load_cnn_model():
    global cnn, classes
    if cnn is None:
        print("Cargando modelo y pesos...")
        cnn = load_model(modelo)
        cnn.load_weights(pesos_modelo)
        print("Modelo y pesos cargados exitosamente.")
        
        print("Cargando datos de clase...")
        data_loader = DataLoader()
        train_generator, _, _, _ = data_loader.load_data()
        class_indices = train_generator.class_indices
        classes = list(class_indices.keys())
        print(f"Clases disponibles: {classes}")

def predict2(file):
    print(f"Procesando archivo: {file}")
    load_cnn_model()
    x = load_img(file, target_size=(longitud, altura))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # Normalizar
    print("Imagen procesada, ejecutando predicción...")
    array = cnn.predict(x)
    result = array[0]
    answer = np.argmax(result)
    predicted_class = classes[answer]
    confidence = result[answer] * 100

    print("Resultados por clase:")
    for i, class_name in enumerate(classes):
        print(f"{class_name}: {result[i] * 100:.2f}%")
    print(f"Predicción: {predicted_class} ({confidence:.2f}% certeza)")

    return predicted_class, confidence

if __name__ == "__main__":
    predict2('./uploaded_images/imgBillete.jpg')
