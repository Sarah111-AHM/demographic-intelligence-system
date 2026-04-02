"""
Emotion Recognition Model Module
Implements 7-class emotion classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, applications
import numpy as np
from typing import Tuple, Dict, Any, List
from sklearn.metrics import classification_report, confusion_matrix

class EmotionRecognizer:
    """Emotion recognition model with 7 emotion classes"""
    
    EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'surprise', 'fear', 'disgust']
    EMOTION_COLORS = {
        'neutral': '#808080',  # Gray
        'happy': '#FFD700',     # Gold
        'sad': '#4169E1',       # Royal Blue
        'angry': '#FF0000',     # Red
        'surprise': '#FFA500',  # Orange
        'fear': '#800080',      # Purple
        'disgust': '#008000'    # Green
    }
    
    def __init__(self, input_shape: Tuple[int, int, int] = (224, 224, 3)):
        self.input_shape = input_shape
        self.model = None
        self.history = None
        
    def build_model(self, trainable_base: bool = False) -> keras.Model:
        """Build emotion recognition model using EfficientNetB0
        
        Args:
            trainable_base: Whether to make base model trainable
            
        Returns:
            Compiled Keras model
        """
        # Use EfficientNetB0 for better accuracy with fewer parameters
        base_model = applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        base_model.trainable = trainable_base
        
        # Build model
        inputs = keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=trainable_base)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(7, activation='softmax', name='emotion')(x)
        
        model = keras.Model(inputs, outputs, name='emotion_recognizer')
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              epochs: int = 50,
              batch_size: int = 32,
              class_weight: Dict = None) -> Dict[str, Any]:
        """Train emotion recognition model
        
        Args:
            X_train: Training images
            y_train: Training labels (0-6)
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
                patience=12,
                mode='max',
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=6,
                min_lr=1e-6
            ),
            keras.callbacks.ModelCheckpoint(
                'models/emotion_model_best.h5',
                monitor='val_accuracy',
                mode='max',
                save_best_only=True
            ),
            keras.callbacks.CSV
