"""
Base classes for face detectors.

Tất cả các detector nên kế thừa từ `BaseDetector` và triển khai
phương thức `detect`. Phương thức này nhận một khung hình (numpy array)
và trả về danh sách bounding box theo định dạng (x, y, w, h).
"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

class BaseAligner(ABC):
    """Abstract base class for face detectors."""

    @abstractmethod
    def align(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Detect faces in a frame.

        Args:
            frame: Ảnh màu BGR dưới dạng numpy array.
            bbox: Bounding box cho khuôn mặt (x, y, w, h), trong đó (x, y) là tọa độ góc trên bên trái,
            w và h là chiều rộng và chiều cao.

        Returns:
            Frame đã được align.
        """
        raise NotImplementedError
    
    def __call__(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Alias cho align() để gọi trực tiếp object như hàm."""
        return self.align(frame, bbox)
