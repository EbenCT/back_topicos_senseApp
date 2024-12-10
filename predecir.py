import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from data_loader import DataLoader

# Configuración
longitud, altura = 150, 150
modelo = './modelo/modelo.keras'
pesos_modelo = './modelo/pesos.weights.h5'

# Cargar el modelo y los pesos
cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)

# Instancia de DataLoader
data_loader = DataLoader()
train_generator, _, _, _ = data_loader.load_data()  # Si ya tienes un método para obtener train_generator

# Obtener las clases desde el generador
class_indices = train_generator.class_indices
classes = list(class_indices.keys())

def predict(file):
    x = load_img(file, target_size=(longitud, altura))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0 
    array = cnn.predict(x)
    result = array[0]
    answer = np.argmax(result)
    predicted_class = classes[answer]
    confidence = result[answer] * 100  # Obtener el porcentaje de certeza

    # Imprimir los porcentajes de todas las clases
    for i, class_name in enumerate(classes):
        print(f"{class_name}: {result[i] * 100:.2f}%")

    print(f"Predicción: {predicted_class} ({confidence:.2f}% certeza)")
    return predicted_class, confidence

# Prueba con una imagen
predict('./uploaded_images/imgBillete.jpg')