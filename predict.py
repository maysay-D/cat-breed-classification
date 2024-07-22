import json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf
from sys import argv

model = tf.keras.models.load_model('model.keras')

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_image(model, image_path, labels):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    top_5_indices = predictions[0].argsort()[-5:][::-1]
    top_5_labels = [labels[i] for i in top_5_indices]
    top_5_scores = [predictions[0][i] for i in top_5_indices]
    return list(zip(top_5_labels, top_5_scores))

with open('class_labels.json', 'r') as f:
    labels = json.load(f)

def main(image_path):
    model = tf.keras.models.load_model('model.keras')
    with open('class_labels.json', 'r') as f:
        labels = json.load(f)
    top_5 = predict_image(model, image_path, labels)
    i = 1
    for label, score in top_5:
        print(f"{label}: {score*100:.2f}%")
        i += 1

if __name__ == "__main__":
    args = argv
    if len(args) > 1:
        image_path = args[1]
        main(image_path)
    else:
        print("Usage: python predict.py <image path>")
