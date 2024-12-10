import os

# Rutas a los directorios de entrenamiento y validación
data_entrenamiento = './data/entrenamiento'
data_validacion = './data/validacion'

# Configuración del entrenamiento
epocas = 20  # Número de épocas para el entrenamiento
longitud, altura = 150, 150  # Dimensiones a las que se redimensionarán las imágenes
batch_size = 32  # Tamaño del lote, es decir, el número de imágenes procesadas en cada paso

def count_images_in_directory(directory):
    """
    Cuenta el número total de imágenes en un directorio, incluyendo subdirectorios.
    """
    total_images = 0
    for subdir, _, files in os.walk(directory):
        total_images += len(files)  # Suma el número de archivos en cada subdirectorio
    return total_images

# Contar el número total de imágenes en los directorios de entrenamiento y validación
total_train_images = count_images_in_directory(data_entrenamiento)
total_val_images = count_images_in_directory(data_validacion)

# Calcular el número de pasos por época
steps_per_epoch = total_train_images // batch_size
validation_steps = total_val_images // batch_size

# Configuración de la red neuronal
filtrosConv1 = 32  # Número de filtros en la primera capa de convolución
filtrosConv2 = 64  # Número de filtros en la segunda capa de convolución
tamano_filtro1 = (3, 3)  # Tamaño del filtro para la primera capa de convolución
tamano_filtro2 = (2, 2)  # Tamaño del filtro para la segunda capa de convolución
tamano_pool = (2, 2)  # Tamaño del filtro de agrupamiento (pooling)

# Contar el número de carpetas en el directorio de entrenamiento para definir el número de clases
clases = len([folder for folder in os.listdir(data_entrenamiento) if os.path.isdir(os.path.join(data_entrenamiento, folder))])

# Tasa de aprendizaje para el optimizador
lr = 0.0004