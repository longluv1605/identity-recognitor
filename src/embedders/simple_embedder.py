from typing import Optional
import cv2
import numpy as np
from .base import BaseEmbedder

class SimpleEmbedder(BaseEmbedder):
    """
    Implementation of BaseEmbedder that creates a simple embedding by
    resizing, grayscaling, flattening and normalising the input image.
    This is only for demonstration; real systems should use deep models.
    """

    def __init__(self, output_dim: Optional[int] = None, size: int = 32):
        """
        Args:
            output_dim: Kích thước vector đầu ra. Nếu None, sẽ bằng size*size.
            size: Kích thước cạnh ảnh chuẩn hoá (size x size).
        """
        super().__init__(output_dim)
        self.size = size
        # Nếu output_dim không chỉ định, dùng size*size
        if self.output_dim is None:
            self.output_dim = size * size

    def embed(self, face: np.ndarray) -> np.ndarray:
        # Nếu ảnh trống, trả về vector zero
        if face.size == 0:
            return np.zeros(self.output_dim, dtype=np.float32)

        # Resize ảnh về kích thước chuẩn
        resized = cv2.resize(face, (self.size, self.size))
        # Chuyển sang grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        # Flatten thành vector
        vector = gray.flatten().astype(np.float32)
        # Normal hoá vector về độ dài 1
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        # Nếu output_dim nhỏ hơn độ dài vector, cắt bớt; nếu lớn hơn, pad thêm 0
        if self.output_dim < vector.size:
            vector = vector[: self.output_dim]
        elif self.output_dim > vector.size:
            pad_width = self.output_dim - vector.size
            vector = np.pad(vector, (0, pad_width), mode="constant")
        return vector
