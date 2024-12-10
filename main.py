# main.py
from data_loader import DataLoader
from model import CNNModel
from config import epocas

def main():
    # Cargar datos
    data_loader = DataLoader()
    train_generator, validation_generator, steps_per_epoch, validation_steps = data_loader.load_data()

    # Crear y compilar el modelo
    cnn_model = CNNModel()
    cnn_model.compile_model()

    # Entrenar el modelo
    cnn_model.train(train_generator, validation_generator, steps_per_epoch, validation_steps, epocas)

    # Guardar el modelo
    cnn_model.save_model()

if __name__ == "__main__":
    main()