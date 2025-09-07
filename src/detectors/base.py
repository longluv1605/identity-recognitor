"""
Base classes for face detectors.

Tất cả các detector nên kế thừa từ `BaseDetector` và triển khai
phương thức `detect`. Phương thức này nhận một khung hình (numpy array)
và trả về danh sách bounding box theo định dạng (x, y, w, h).
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

class BaseDetector(ABC):
    """Abstract base class for face detectors."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a frame.

        Args:
            frame: Ảnh màu BGR dưới dạng numpy array.

        Returns:
            Danh sách các bounding box, mỗi box được biểu diễn bởi tuple
            (x, y, w, h), trong đó (x, y) là tọa độ góc trên bên trái,
            w và h là chiều rộng và chiều cao.
        """
        raise NotImplementedError
    
    
    def __call__(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Alias cho detect() để gọi trực tiếp object như hàm."""
        return self.detect(frame)
