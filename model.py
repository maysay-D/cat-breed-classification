import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json


def load_data(image_dir):
    breeds = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
    data = []
    labels = []
    for breed in breeds:
        breed_dir = os.path.join(image_dir, breed)
        images = [os.path.join(breed_dir, img) for img in os.listdir(breed_dir)]
        for img_path in images:
            data.append(img_path)
            labels.append(breed)
    return data, labels, breeds


def split_data(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, stratify=labels)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train)
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_data_generators(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32):
    datagen_train = ImageDataGenerator(rescale=1. / 255, rotation_range=20, width_shift_range=0.2,
                                       height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                       brightness_range=[0.8,1.2], channel_shift_range=50, 
                                       vertical_flip=True, horizontal_flip=True, fill_mode='nearest')
    datagen_val_test = ImageDataGenerator(rescale=1. / 255)

    train_df = pd.DataFrame({'filename': X_train, 'class': y_train})
    val_df = pd.DataFrame({'filename': X_val, 'class': y_val})
    test_df = pd.DataFrame({'filename': X_test, 'class': y_test})

    train_generator = datagen_train.flow_from_dataframe(train_df,
                                                        x_col='filename', y_col='class', target_size=(224, 224),
                                                        batch_size=batch_size, class_mode='categorical')
    val_generator = datagen_val_test.flow_from_dataframe(val_df,
                                                         x_col='filename', y_col='class', target_size=(224, 224),
                                                         batch_size=batch_size, class_mode='categorical')
    test_generator = datagen_val_test.flow_from_dataframe(test_df,
                                                          x_col='filename', y_col='class', target_size=(224, 224),
                                                          batch_size=batch_size, class_mode='categorical')

    class_labels = train_generator.class_indices
    breed_labels = list(class_labels.keys())
    return train_generator, val_generator, test_generator, breed_labels


def create_model(num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    return model


def train_model(model, train_generator, val_generator, epochs=30):
    callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=[callback]
    )

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return history


def evaluate_model(model, test_generator, breed_labels):
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_generator.classes

    cm = confusion_matrix(y_true, y_pred_classes)

    plt.figure(figsize=(10, 7))
    plt.title('Confusion Matrix')
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()

    tick_marks = np.arange(len(breed_labels))
    plt.xticks(tick_marks, breed_labels, rotation=90)
    plt.yticks(tick_marks, breed_labels)

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()


def save_model(model, breed_labels, filename='model.keras', class_filename='class_labels.json'):
    model.save(filename)

    with open(class_filename, 'w') as f:
        json.dump(breed_labels, f)

def main(image_dir):
    data, labels, breeds = load_data(image_dir)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data, labels)
    train_generator, val_generator, test_generator, breed_labels = create_data_generators(X_train, X_val, X_test, y_train, y_val, y_test)

    num_classes = len(breed_labels)
    model = create_model(num_classes)
    
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    history = train_model(model, train_generator, val_generator, epochs=30)
    evaluate_model(model, test_generator, breed_labels)

    save_model(model, breed_labels)


if __name__ == "__main__":
    image_dir = 'images'
    main(image_dir)
