"""
Gender Classification Model Module
Implements binary classification for gender prediction
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, applications
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class GenderClassifier:
    """Gender classification model using MobileNetV2 for efficiency"""
    
    def __init__(self, input_shape: Tuple[int, int, int] = (224, 224, 3)):
        self.input_shape = input_shape
        self.model = None
        self.history = None
        
    def build_model(self, trainable_base: bool = False) -> keras.Model:
        """Build gender classification model
        
        Args:
            trainable_base: Whether to make base model trainable
            
        Returns:
            Compiled Keras model
        """
        # Use MobileNetV2 for faster inference
        base_model = applications.MobileNetV2(
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
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(1, activation='sigmoid', name='gender')(x)
        
        model = keras.Model(inputs, outputs, name='gender_classifier')
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        self.model = model
        return model
    
    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              epochs: int = 40,
              batch_size: int = 32,
              class_weight: Dict = None) -> Dict[str, Any]:
        """Train gender classification model
        
        Args:
            X_train: Training images
            y_train: Training labels (0: male, 1: female)
            X_val: Validation images
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            class_weight: Optional class weights for imbalance
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                mode='max',
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            keras.callbacks.ModelCheckpoint(
                'models/gender_model_best.h5',
                monitor='val_accuracy',
                mode='max',
                save_best_only=True
            )
        ]
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )
        
        return self.history.history
    
    def predict(self, images: np.ndarray) -> np.ndarray:
        """Predict gender probabilities
        
        Args:
            images: Input images array
            
        Returns:
            Predicted probabilities (0-1)
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")
        
        predictions = self.model.predict(images)
        return predictions.flatten()
    
    def predict_class(self, images: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict gender classes
        
        Args:
            images: Input images array
            threshold: Classification threshold
            
        Returns:
            Predicted classes (0: male, 1: female)
        """
        probabilities = self.predict(images)
        return (probabilities >= threshold).astype(int)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance with multiple metrics
        
        Args:
            X_test: Test images
            y_test: True labels
            
        Returns:
            Dictionary of metrics
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")
        
        # Get predictions
        y_pred_proba = self.predict(X_test)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Also get model's own evaluation
        loss, accuracy, precision, recall = self.model.evaluate(X_test, y_test, verbose=0)
        metrics['model_loss'] = loss
        metrics['model_accuracy'] = accuracy
        
        return metrics
    
    def load_model(self, model_path: str):
        """Load pre-trained model"""
        self.model = keras.models.load_model(model_path)
        print(f"✅ Model loaded from {model_path}")
    
    def get_gender_confidence(self, images: np.ndarray) -> Dict[str, np.ndarray]:
        """Get confidence scores for predictions
        
        Args:
            images: Input images
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        probabilities = self.predict(images)
        
        return {
            'predictions': (probabilities >= 0.5).astype(int),
            'male_confidence': 1 - probabilities,
            'female_confidence': probabilities
        }

# Example usage with data augmentation
class GenderDataAugmentation:
    """Custom data augmentation for gender classification"""
    
    def __init__(self):
        self.augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
            layers.RandomBrightness(0.1)
        ])
    
    def augment(self, images: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply augmentation to images"""
        augmented_images = self.augmentation(images, training=True)
        return augmented_images.numpy(), labels

if __name__ == "__main__":
    # Test the model
    classifier = GenderClassifier()
    model = classifier.build_model()
    model.summary()
