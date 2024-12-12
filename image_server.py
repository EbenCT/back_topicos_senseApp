from flask import Flask, request, jsonify
import os
from predecir2 import predict2  # Importar la función predict de predecir.py

app = Flask(__name__)

# Define el directorio donde se guardarán las imágenes
UPLOAD_FOLDER = 'uploaded_images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    print(f"Directorio '{UPLOAD_FOLDER}' creado.")

@app.route('/upload_image', methods=['POST'])
def upload_image():
    print("Solicitud POST recibida en '/upload_image'")
    if 'image' not in request.files:
        print("Error: No se encontró el archivo 'image' en la solicitud.")
        return 'No image part', 400

    file = request.files['image']
    if file.filename == '':
        print("Error: No se seleccionó ningún archivo.")
        return 'No selected file', 400

    # Guardar la imagen con el nombre "imgBillete.jpg", sobrescribiendo la anterior si ya existe
    filename = "imgBillete.jpg"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    print(f"Imagen guardada en: {file_path}")

    try:
        # Llamar a la función predict y obtener la predicción
        print("Ejecutando predicción con 'predict2'...")
        predicted_class, confidence = predict2(file_path)
        print(f"Predicción completada: {predicted_class} con {confidence:.2f}% de confianza.")
    except Exception as e:
        print(f"Error durante la predicción: {e}")
        return jsonify({"error": "Error interno durante la predicción"}), 500

    # Crear la respuesta JSON
    response = {
        'predicted_class': predicted_class,
        'confidence': confidence
    }
    print("Respuesta generada:", response)

    # Devolver la predicción y la confianza en formato JSON
    return jsonify(response), 200

if __name__ == '__main__':
    print("Iniciando servidor Flask en modo desarrollo...")
    app.run(host='0.0.0.0', port=5000)
