# model.py
import os  
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dropout, Flatten, Dense, Convolution2D, MaxPooling2D  
from tensorflow.keras.optimizers import Adam  
from config import longitud, altura, filtrosConv1, filtrosConv2, tamano_filtro1, tamano_filtro2, tamano_pool, clases, lr 

class CNNModel:
    def __init__(self):
        """
        Inicializa el modelo de red neuronal convolucional.
        """
        self.model = Sequential()  # Crea un modelo secuencial

        # Añade la primera capa convolucional con activación ReLU
        self.model.add(Convolution2D(filtrosConv1, tamano_filtro1, padding="same", input_shape=(longitud, altura, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=tamano_pool))  # Añade una capa de agrupamiento (pooling)

        # Añade la segunda capa convolucional con activación ReLU
        self.model.add(Convolution2D(filtrosConv2, tamano_filtro2, padding="same", activation='relu'))
        self.model.add(MaxPooling2D(pool_size=tamano_pool))  # Añade otra capa de agrupamiento (pooling)

        self.model.add(Flatten())  # Aplana las salidas de las capas convolucionales
        self.model.add(Dense(256, activation='relu'))  # Añade una capa densa (fully connected) con activación ReLU
        self.model.add(Dropout(0.5))  # Añade una capa de abandono (dropout) para prevenir el sobreajuste
        self.model.add(Dense(clases, activation='softmax'))  # Añade la capa de salida con activación softmax para clasificación múltiple

    def compile_model(self):
        """
        Compila el modelo con una función de pérdida, un optimizador y una métrica.
        """
        self.model.compile(loss='categorical_crossentropy',  # Función de pérdida para clasificación múltiple
                           optimizer=Adam(learning_rate=lr),  # Optimizador Adam con tasa de aprendizaje especificada
                           metrics=['accuracy'])  # Métrica de precisión

    def train(self, train_generator, validation_generator, steps_per_epoch, validation_steps, epochs):
        """
        Entrena el modelo utilizando generadores de datos para entrenamiento y validación.
        
        :param train_generator: Generador de datos para el conjunto de entrenamiento.
        :param validation_generator: Generador de datos para el conjunto de validación.
        :param steps_per_epoch: Número de pasos por época.
        :param validation_steps: Número de pasos para validación.
        :param epochs: Número de épocas para el entrenamiento.
        """
        self.model.fit(
            train_generator,  # Generador de datos para el entrenamiento
            steps_per_epoch=steps_per_epoch,  # Número de pasos por época
            epochs=epochs,  # Número de épocas
            validation_data=validation_generator,  # Generador de datos para la validación
            validation_steps=validation_steps)  # Número de pasos para la validación

    def save_model(self, model_path='./modelo/modelo.keras', weights_path='./modelo/pesos.weights.h5'):
        """
        Guarda el modelo y sus pesos en los archivos especificados.
        
        :param model_path: Ruta del archivo para guardar el modelo.
        :param weights_path: Ruta del archivo para guardar los pesos del modelo.
        """
        if not os.path.exists('./modelo/'):  # Verifica si el directorio del modelo no existe
            os.mkdir('./modelo/')  # Crea el directorio del modelo
        self.model.save(model_path)  # Guarda el modelo completo
        self.model.save_weights(weights_path)  # Guarda solo los pesos del modelo