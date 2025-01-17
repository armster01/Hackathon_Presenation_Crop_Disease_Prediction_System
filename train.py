import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import CropDiseaseModel
import os

def prepare_data():
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    # Only rescaling for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    # Load and prepare the training data
    train_generator = train_datagen.flow_from_directory(
        'dataset/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    # Load and prepare the validation data
    validation_generator = val_datagen.flow_from_directory(
        'dataset/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, validation_generator

def train_model():
    # Create model instance
    model = CropDiseaseModel()
    
    # Prepare data
    train_generator, validation_generator = prepare_data()
    
    # Train the model
    history = model.train(
        train_generator,
        validation_data=validation_generator,
        epochs=20
    )
    
    # Save the trained weights
    if not os.path.exists('model'):
        os.makedirs('model')
    model.save_weights('model/weights.h5')
    
    return history

if __name__ == '__main__':
    history = train_model()