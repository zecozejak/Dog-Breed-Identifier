import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # Ignorowanie uszkodzonych obrazów

# Ustawienia ścieżek do katalogów
base_dir = 'dogImages'
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'valid')
test_dir = os.path.join(base_dir, 'test')

# Parametry modelu
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32


# Funkcja do weryfikacji obrazów
def is_valid_image(filepath):
    try:
        img = Image.open(filepath)
        img.verify()
        img.close()
        return True
    except (IOError, SyntaxError):
        return False


# Funkcja do filtrowania katalogów
def filter_valid_images(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if not is_valid_image(file_path):
                print(f'Removing invalid image: {file_path}')
                os.remove(file_path)


# Filtruj obrazy w katalogach
filter_valid_images(train_dir)
filter_valid_images(valid_dir)
filter_valid_images(test_dir)

# ImageDataGenerator do przeskalowania obrazów
train_image_generator = ImageDataGenerator(rescale=1. / 255)
validation_image_generator = ImageDataGenerator(rescale=1. / 255)
test_image_generator = ImageDataGenerator(rescale=1. / 255)

# Generatory danych
train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='categorical')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                              directory=valid_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='categorical')

test_data_gen = test_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                         directory=test_dir,
                                                         target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                         class_mode='categorical')

# Konwersja generatorów danych na tf.data.Dataset
train_dataset = tf.data.Dataset.from_generator(lambda: train_data_gen, output_signature=(
    tf.TensorSpec(shape=(None, IMG_HEIGHT, IMG_WIDTH, 3), dtype=tf.float32),
    tf.TensorSpec(shape=(None, 133), dtype=tf.float32)))
train_dataset = train_dataset.repeat().prefetch(tf.data.AUTOTUNE)

valid_dataset = tf.data.Dataset.from_generator(lambda: val_data_gen, output_signature=(
    tf.TensorSpec(shape=(None, IMG_HEIGHT, IMG_WIDTH, 3), dtype=tf.float32),
    tf.TensorSpec(shape=(None, 133), dtype=tf.float32)))
valid_dataset = valid_dataset.repeat().prefetch(tf.data.AUTOTUNE)

# Tworzenie modelu z wykorzystaniem MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Zablokowanie warstw pretrenowanego modelu
for layer in base_model.layers:
    layer.trainable = False

# Dodanie własnych warstw
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.4)(x)
predictions = Dense(133, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Kompilacja modelu
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Trenowanie modelu
epochs = 25

# Zmiana rozszerzenia pliku na .keras
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.keras',
                               verbose=1, save_best_only=True)

history = model.fit(train_dataset,
                    validation_data=valid_dataset,
                    epochs=epochs,
                    steps_per_epoch=len(train_data_gen),
                    validation_steps=len(val_data_gen),
                    callbacks=[checkpointer],
                    verbose=1)

# Zapisanie całego modelu
model.save('dog_breed_classifier.keras')

# Ocena modelu
# Załaduj wcześniej zapisany model
model = tf.keras.models.load_model('dog_breed_classifier.keras')

# Predykcja na zbiorze testowym
test_images, test_labels = next(test_data_gen)  # to obtain the test images and labels
dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_images]

# Raportowanie dokładności
test_accuracy = 100 * np.sum(np.array(dog_breed_predictions) == np.argmax(test_labels, axis=1)) / len(
    dog_breed_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
