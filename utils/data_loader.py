"""
Data Loading Utilities for UTKFace Dataset
Handles downloading, loading, and preprocessing of the dataset
"""

import os
import numpy as np
import pandas as pd
import cv2
import requests
import tarfile
from tqdm import tqdm
from typing import Tuple, Optional, List, Dict
import zipfile
from pathlib import Path

class UTKFaceLoader:
    """Loader for UTKFace dataset with automatic download"""
    
    def __init__(self, data_dir: str = 'data/UTKFace/'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.images = []
        self.labels = []
        
    def download_dataset(self, url: str = None):
        """Download UTKFace dataset if not already present"""
        
        if url is None:
            # Note: This is a placeholder URL. In production, use official UTKFace source
            url = "https://www.kaggle.com/datasets/jangedoo/utkface-new"
        
        print("⚠️ Please download UTKFace dataset manually from:")
        print("   https://www.kaggle.com/datasets/jangedoo/utkface-new")
        print(f"   and place the images in {self.data_dir}")
        
        # Alternative: Use a smaller subset from another source
        self._download_sample_subset()
    
    def _download_sample_subset(self):
        """Download a sample subset for demonstration"""
        
        # Create sample data (for demonstration)
        print("Creating sample dataset for demonstration...")
        
        # Generate synthetic samples
        for i in range(100):
            # Create blank image
            img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
            
            # Generate random label
            age = np.random.randint(1, 100)
            gender = np.random.randint(0, 2)
            race = np.random.randint(0, 5)
            
            # Save image
            filename = f"{age}_{gender}_{race}_{i}.jpg"
            cv2.imwrite(str(self.data_dir / filename), img)
        
        print(f"✅ Created {len(list(self.data_dir.glob('*.jpg')))} sample images")
    
    def load_dataset(self, 
                     max_samples: Optional[int] = None,
                     target_size: Tuple[int, int] = (224, 224)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load and preprocess UTKFace dataset
        
        Args:
            max_samples: Maximum number of samples to load
            target_size: Target image size (height, width)
            
        Returns:
            Tuple of (images, ages, genders)
        """
        
        print(f"Loading dataset from {self.data_dir}...")
        
        image_files = list(self.data_dir.glob('*.jpg'))
        
        if max_samples:
            image_files = image_files[:max_samples]
        
        images = []
        ages = []
        genders = []
        
        for img_path in tqdm(image_files, desc="Loading images"):
            try:
                # Parse filename: [age]_[gender]_[race]_[date].jpg
                filename = img_path.stem
                parts = filename.split('_')
                
                if len(parts) >= 2:
                    age = int(parts[0])
                    gender = int(parts[1])
                    
                    # Load and preprocess image
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, target_size)
                        
                        images.append(img)
                        ages.append(age)
                        genders.append(gender)
                        
            except Exception as e:
                continue
        
        images = np.array(images, dtype=np.float32)
        ages = np.array(ages, dtype=np.int32)
        genders = np.array(genders, dtype=np.int32)
        
        print(f"✅ Loaded {len(images)} images")
        print(f"   Age range: {ages.min()} - {ages.max()}")
        print(f"   Gender distribution: Male={(genders==0).sum()}, Female={(genders==1).sum()}")
        
        return images, ages, genders
    
    def create_dataframe(self, images: np.ndarray, ages: np.ndarray, genders: np.ndarray) -> pd.DataFrame:
        """Create pandas DataFrame for analysis"""
        
        df = pd.DataFrame({
            'age': ages,
            'gender': ['Male' if g == 0 else 'Female' for g in genders],
            'gender_code': genders
        })
        
        # Add age groups
        df['age_group'] = pd.cut(df['age'], 
                                 bins=[0, 18, 30, 50, 65, 100],
                                 labels=['Child', 'Young Adult', 'Adult', 'Middle Age', 'Senior'])
        
        return df

class DataAugmentation:
    """Data augmentation utilities for training"""
    
    def __init__(self):
        import tensorflow as tf
        self.augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
            tf.keras.layers.RandomBrightness(0.1),
        ])
    
    def augment_batch(self, images: np.ndarray) -> np.ndarray:
        """Apply augmentation to a batch of images"""
        import tensorflow as tf
        augmented = self.augmentation(images, training=True)
        return augmented.numpy()

# Example usage
if __name__ == "__main__":
    loader = UTKFaceLoader()
    images, ages, genders = loader.load_dataset(max_samples=100)
    df = loader.create_dataframe(images, ages, genders)
    print(df.head())
    print(df['age_group'].value_counts())
