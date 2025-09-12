import numpy as np
from deepface import DeepFace
import cv2
import tempfile
import os
import uuid
from .base import BaseEmbedder


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Row / vector wise L2 normalisation."""
    n = np.linalg.norm(vec)
    if n > 0:
        return vec / n
    return vec

class DeepFaceEmbedder(BaseEmbedder):

    def __init__(self,
                 model_name: str = "Facenet512",
                 enforce_detection: bool = False,
                 normalize: bool = True):
        """DeepFace Embedder."""
        self.model_name = model_name
        self.enforce_detection = enforce_detection
        self.normalize = normalize

        # Xác định output dimension
        self.output_dims = {
            'VGG-Face': 2622,
            'Facenet': 128,
            'Facenet512': 512,
            'OpenFace': 128,
            'DeepFace': 4096,
            'DeepID': 160,
            'ArcFace': 512,
            'Dlib': 128,
            'SFace': 128
        }

        self.output_dim = self.output_dims.get(model_name, 512)
        print(f"Initialized DeepFace with {model_name} (dim={self.output_dim}) | normalize={self.normalize}")
        
    
    def embed(self, face: np.ndarray) -> np.ndarray:
        """
        Trích xuất embedding từ ảnh khuôn mặt
        
        Args:
            face: Ảnh khuôn mặt BGR (H, W, 3)
            
        Returns:
            Embedding vector (output_dim,)
        """
        temp_path = None
        try:
            # DeepFace yêu cầu file path, nên ta phải save temp file
            # Tạo temp file path unique
            import uuid
            temp_dir = tempfile.gettempdir()
            temp_filename = f"deepface_temp_{uuid.uuid4().hex}.jpg"
            temp_path = os.path.join(temp_dir, temp_filename)
            
            # Save image
            cv2.imwrite(temp_path, face)
            
            # Trích xuất embedding
            embedding_objs = DeepFace.represent(
                img_path=temp_path,
                model_name=self.model_name,
                enforce_detection=self.enforce_detection,
                detector_backend='skip'  # we already provide aligned face
            )
            
            # DeepFace trả về list of dict
            if len(embedding_objs) > 0:
                embedding = np.array(embedding_objs[0]["embedding"], dtype=np.float32)
            else:
                embedding = np.zeros(self.output_dim, dtype=np.float32)

            if self.normalize:
                embedding = _l2_normalize(embedding)

            return embedding
                    
        except Exception as e:
            print(f"Error in DeepFace embedding: {e}")
            out = np.zeros(self.output_dim, dtype=np.float32)
            if self.normalize:
                return out  # already zero
            return out
            
        finally:
            # Cleanup temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass  # Ignore cleanup errors