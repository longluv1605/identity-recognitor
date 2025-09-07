from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseEmbedder(ABC):
    """Abstract base class for face embedders."""

    def __init__(self, output_dim: Optional[int] = None):
        """
        Args:
            output_dim: Số chiều của vector embedding mà lớp con sẽ trả về.
        """
        self.output_dim = output_dim

    @abstractmethod
    def embed(self, face: np.ndarray) -> np.ndarray:
        """
        Trích xuất đặc trưng từ một khuôn mặt đã căn chỉnh.

        Args:
            face: Ảnh khuôn mặt (numpy array BGR hoặc RGB tuỳ mô hình).

        Returns:
            Vector numpy 1D biểu diễn khuôn mặt.
        """
        raise NotImplementedError

    def __call__(self, face: np.ndarray) -> np.ndarray:
        """Cho phép gọi embedder như một hàm."""
        return self.embed(face)