"""
Image Preprocessing Utilities
Face detection, alignment, and normalization
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
from mtcnn import MTCNN
import albumentations as A

class FacePreprocessor:
    """Handles face detection, alignment, and preprocessing"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        self.detector = MTCNN()  # More accurate face detection
        
    def detect_faces(self, image: np.ndarray) -> List[dict]:
        """Detect faces in image using MTCNN
        
        Args:
            image: Input image (RGB)
            
        Returns:
            List of face dictionaries with bounding boxes and landmarks
        """
        # MTCNN expects RGB image
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        faces = self.detector.detect_faces(image)
        return faces
    
    def extract_face(self, 
                     image: np.ndarray, 
                     bounding_box: Tuple[int, int, int, int],
                     margin: float = 0.2) -> np.ndarray:
        """Extract and align face from bounding box
        
        Args:
            image: Input image
            bounding_box: (x, y, width, height)
            margin: Margin around face (percentage)
            
        Returns:
            Cropped and aligned face
        """
        x, y, w, h = bounding_box
        
        # Add margin
        margin_x = int(w * margin)
        margin_y = int(h * margin)
        
        x = max(0, x - margin_x)
        y = max(0, y - margin_y)
        w = min(image.shape[1] - x, w + 2 * margin_x)
        h = min(image.shape[0] - y, h + 2 * margin_y)
        
        # Extract face
        face = image[y:y+h, x:x+w]
        
        # Resize to target size
        face = cv2.resize(face, self.target_size)
        
        return face
    
    def preprocess_face(self, 
                        face: np.ndarray,
                        normalize: bool = True) -> np.ndarray:
        """Preprocess extracted face for model input
        
        Args:
            face: Face image
            normalize: Whether to normalize pixel values
            
        Returns:
            Preprocessed face
        """
        # Convert to RGB if needed
        if len(face.shape) == 2:
            face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        elif face.shape[2] == 4:
            face = cv2.cvtColor(face, cv2.COLOR_RGBA2RGB)
        
        # Resize to target size
        face = cv2.resize(face, self.target_size)
        
        # Normalize
        if normalize:
            face = face.astype(np.float32) / 255.0
        
        return face
    
    def process_image(self, 
                      image: np.ndarray,
                      max_faces: int = 5) -> Tuple[List[np.ndarray], List[dict]]:
        """Complete face processing pipeline
        
        Args:
            image: Input image
            max_faces: Maximum number of faces to extract
            
        Returns:
            Tuple of (processed faces, face metadata)
        """
        # Detect faces
        faces = self.detect_faces(image)
        
        # Sort by confidence and limit
        faces = sorted(faces, key=lambda x: x['confidence'], reverse=True)
        faces = faces[:max_faces]
        
        processed_faces = []
        face_metadata = []
        
        for face_data in faces:
            bbox = face_data['box']
            face = self.extract_face(image, bbox)
            processed_face = self.preprocess_face(face)
            
            processed_faces.append(processed_face)
            face_metadata.append({
                'bbox': bbox,
                'confidence': face_data['confidence'],
                'keypoints': face_data.get('keypoints', {})
            })
        
        return processed_faces, face_metadata

class ImageNormalizer:
    """Image normalization utilities"""
    
    @staticmethod
    def z_score_normalize(image: np.ndarray) -> np.nd
