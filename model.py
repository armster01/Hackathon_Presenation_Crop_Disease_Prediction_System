import tensorflow as tf
from tensorflow.keras import layers, models

class CropDiseaseModel:
    def __init__(self):
        self.model = self._build_model()
    
    def _build_model(self):
        model = models.Sequential([
            # Input layer
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            layers.MaxPooling2D((2, 2)),
            
            # Convolutional layers
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(8, activation='softmax')  # 8 classes for different diseases
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, train_data, validation_data, epochs=10):
        return self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs
        )
    
    def predict(self, image):
        return self.model.predict(image)
    
    def load_weights(self, path):
        self.model.load_weights(path)
    
    def save_weights(self, path):
        self.model.save_weights(path)