"""
Age Prediction Model Module
Implements CNN-based age regression using transfer learning
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, applications
import numpy as np
from typing import Tuple, Dict, Any

class AgePredictor:
    """Age prediction model using ResNet50 backbone"""
    
    def __init__(self, input_shape: Tuple[int, int, int] = (224, 224, 3)):
        self.input_shape = input_shape
        self.model = None
        self.history = None
        
    def build_model(self, trainable_base: bool = False) -> keras.Model:
        """Build age prediction model
        
        Args:
            trainable_base: Whether to make base model trainable
            
        Returns:
            Compiled Keras model
        """
        # Base model
        base_model = applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        base_model.trainable = trainable_base
        
        # Build model
        inputs = keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=trainable_base)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(1, name='age')(x)
        
        model = keras.Model(inputs, outputs, name='age_predictor')
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss='mae',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              epochs: int = 50,
              batch_size: int = 32) -> Dict[str, Any]:
        """Train age prediction model
        
        Args:
            X_train: Training images
            y_train: Training ages
            X_val: Validation images
            y_val: Validation ages
            epochs: Number of epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            keras.callbacks.ModelCheckpoint(
                'models/age_model_best.h5',
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history.history
    
    def predict(self, images: np.ndarray) -> np.ndarray:
        """Predict ages for input images
        
        Args:
            images: Input images array
            
        Returns:
            Predicted ages
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")
        
        predictions = self.model.predict(images)
        return predictions.flatten()
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance
        
        Args:
            X_test: Test images
            y_test: True ages
            
        Returns:
            Dictionary of metrics
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")
        
        loss, mae = self.model.evaluate(X_test, y_test, verbose=0)
        
        return {
            'loss': loss,
            'mae': mae
        }
    
    def load_model(self, model_path: str):
        """Load pre-trained model"""
        self.model = keras.models.load_model(model_path)
        print(f"✅ Model loaded from {model_path}")
    
    def fine_tune(self, 
                  X_train: np.ndarray,
                  y_train: np.ndarray,
                  X_val: np.ndarray,
                  y_val: np.ndarray,
                  epochs: int = 30,
                  learning_rate: float = 1e-5):
        """Fine-tune the model with unfrozen base layers"""
        
        # Unfreeze base model
        for layer in self.model.layers[1].layers[:100]:  # ResNet50 is second layer
            layer.trainable = True
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mae',
            metrics=['mae']
        )
        
        # Continue training
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            verbose=1
        )
        
        return history.history

# Example usage
if __name__ == "__main__":
    # Test the model
    predictor = AgePredictor()
    model = predictor.build_model()
    model.summary()
