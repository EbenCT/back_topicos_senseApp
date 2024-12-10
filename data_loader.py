from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from config import data_entrenamiento, data_validacion, longitud, altura, batch_size, steps_per_epoch, validation_steps

class DataLoader:
    def __init__(self):
        # Inicializa el generador de datos para el entrenamiento con aumento de datos
        self.train_datagen = ImageDataGenerator(
            rescale=1. / 255, 
            shear_range=0.2,  
            zoom_range=0.2,  
            horizontal_flip=True)  

        # Inicializa el generador de datos para la validación sin aumento de datos
        self.test_datagen = ImageDataGenerator(rescale=1. / 255)  

    def load_data(self):
        # Crea un generador de datos para el conjunto de entrenamiento
        train_generator = self.train_datagen.flow_from_directory(
            data_entrenamiento,  
            target_size=(altura, longitud),  
            batch_size=batch_size, 
            class_mode='categorical')

        # Crea un generador de datos para el conjunto de validación
        validation_generator = self.test_datagen.flow_from_directory(
            data_validacion,
            target_size=(altura, longitud),
            batch_size=batch_size,
            class_mode='categorical')

        # Devuelve los generadores de datos y los pasos calculados
        return train_generator, validation_generator, steps_per_epoch, validation_steps