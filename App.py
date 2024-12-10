import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model

# Set up environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load annotations
def load_annotations(csv_file):
    df = pd.read_csv(csv_file)
    return df

# Function to load and preprocess an image
def load_image(file_path):
    img = cv2.imread(file_path)
    img = cv2.resize(img, (416, 416))
    return img

# Create dataset
def create_dataset(df, num_samples=20000):
    images = []
    boxes = []
    labels = []

    for _, row in df.head(num_samples).iterrows():
        img_path = os.path.join('test', row['filename'])
        img = load_image(img_path)
        
        # Ensure x_min, y_min, x_max, y_max are defined
        x_min, y_min, x_max, y_max = row[2:]
        box = np.array([x_min, y_min, x_max, y_max])
        label = row['class']  # Keep the class as a string
        
        images.append(img)
        boxes.append(box)
        labels.append(label)

    return np.array(images), np.array(boxes), np.array(labels)

# Normalize coordinates
def normalize_coords(boxes):
    return np.array([
        (boxes[:2] + boxes[2:]) / 416,
        (boxes[2:] - boxes[:2]) / 416
    ]).T

# Create model
def create_model(input_shape=(416, 416, 3), num_classes=None):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# Create data generators
def create_data_generators(train_images, train_boxes, train_labels):
    train_datagen = ImageDataGenerator(
        shear_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=30,
        validation_split=0.2
    )

    val_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(
        train_images,
        train_boxes,
        sample_weight=np.ones_like(train_labels),
        batch_size=32,
        subset='training'
    )

    val_generator = val_datagen.flow(
        train_images[train_images.shape[0]//5:],
        train_boxes[train_images.shape[0]//5:],
        batch_size=32,
        subset='validation'
    )

    return train_generator, val_generator

# Main function
def main():
    # Load annotations
    df = load_annotations('train/_annotations.csv')

    # Create dataset
    images, boxes, labels = create_dataset(df, num_samples=20000)

    # Normalize coordinates
    normalized_boxes = normalize_coords(boxes)

    # Split data into training and validation sets
    train_images, val_images, train_boxes, val_boxes, train_labels, val_labels = train_test_split(
        images, normalized_boxes, labels, test_size=0.2, random_state=42
    )

    # Create model
    input_shape = (416, 416, 3)
    num_classes = len(set(labels))
    model = create_model(input_shape, num_classes)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Create data generators
    train_generator, val_generator = create_data_generators(train_images, train_boxes, train_labels)

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // 32,
        validation_data=val_generator,
        validation_steps=val_generator.samples // 32,
        epochs=50,
        verbose=1,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)]
    )

    # Evaluate the model on validation set
    loss, accuracy = model.evaluate(val_generator, steps=val_generator.samples // 32)
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")

    # Save the trained model
    model.save('shape_recognition_model.h5')

    # Plot model architecture
    plot_model(model, to_file='model_architecture.png', show_shapes=True)

if __name__ == "__main__":
    main()
