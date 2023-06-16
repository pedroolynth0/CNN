from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# Inicializando a Rede Neural Convolucional
classifier = Sequential()

# Passo 1 - Primeira Camada de Convolução
classifier.add(Conv2D(64, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# Passo 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adicionando a Segunda Camada de Convolução
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adicionando a Terceira Camada de Convolução
classifier.add(Conv2D(128, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Passo 3 - Flattening
classifier.add(Flatten())

# Passo 4 - Full connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

# Compilando a rede
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Criando os objetos train_datagen e validation_datagen com as regras de pré-processamento das imagens
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Pré-processamento das imagens de treino e validação
training_set = train_datagen.flow_from_directory('dataset_treino',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

# Defina class_mode como None se você não tiver um conjunto de validação
validation_set = validation_datagen.flow_from_directory('dataset_validation',
                                                        target_size=(64, 64),
                                                        batch_size=32,
                                                        class_mode=None)

# Executando o treinamento
classifier.fit(training_set,
               steps_per_epoch=len(training_set),
               epochs=20,
               validation_data=validation_set,
               validation_steps=len(validation_set))

# Carregando e fazendo a previsão das imagens de teste
test_image = keras.preprocessing.image.load_img('dataset_teste/bart10.bmp', target_size=(64, 64))
test_image = keras.preprocessing.image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
    prediction = 'Homer'
else:
    prediction = 'Bart'

# Previsão da primeira imagem
print("Correto: Bart")
print(prediction)

# Carregando e fazendo a previsão das imagens de teste
test_image = keras.preprocessing.image.load_img('dataset_teste/bart4.bmp', target_size=(64, 64))
test_image = keras.preprocessing.image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
    prediction = 'Homer'
else:
    prediction = 'Bart'

# Previsão da primeira imagem
print("Correto: Bart")
print(prediction)

# Carregando e fazendo a previsão das imagens de teste
test_image = keras.preprocessing.image.load_img('dataset_teste/bart17.bmp', target_size=(64, 64))
test_image = keras.preprocessing.image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
    prediction = 'Homer'
else:
    prediction = 'Bart'

# Previsão da primeira imagem
print("Correto: Bart")
print(prediction)

# Segunda Imagem
test_image = keras.preprocessing.image.load_img('dataset_teste/homer9.bmp', target_size=(64, 64))
test_image = keras.preprocessing.image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
    prediction = 'Homer'
else:
    prediction = 'Bart'

print("Correto: Homer")

# Previsão da segunda imagem
print(prediction)

# Terceira Imagem
test_image = keras.preprocessing.image.load_img('dataset_teste/homer22.bmp', target_size=(64, 64))
test_image = keras.preprocessing.image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
    prediction = 'Homer'
else:
    prediction = 'Bart'

print("Correto: Homer")
# Previsão da terceira imagem
print(prediction)
