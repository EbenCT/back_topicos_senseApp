from flask import Flask, request, jsonify
import os
from predecir2 import predict2  # Importar la función predict de predecir.py

app = Flask(__name__)

# Define el directorio donde se guardarán las imágenes
UPLOAD_FOLDER = 'uploaded_images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return 'No image part', 400

    file = request.files['image']
    if file.filename == '':
        return 'No selected file', 400

    # Guardar la imagen con el nombre "imgBillete.jpg", sobrescribiendo la anterior si ya existe
    filename = "imgBillete.jpg"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # Llamar a la función predict y obtener la predicción
    predicted_class, confidence = predict2(file_path)

    # Crear la respuesta JSON
    response = {
        'predicted_class': predicted_class,
        'confidence': confidence
    }

    # Imprimir la respuesta JSON en consola
    print("Predicción:", response)

    # Devolver la predicción y la confianza en formato JSON
    return jsonify(response), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
