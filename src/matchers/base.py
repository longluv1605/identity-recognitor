"""
Matcher base class and simple implementation.

Matcher nhận một vector embedding và so sánh với cơ sở dữ liệu embeddings
để tìm ra người phù hợp nhất. Tùy vào kích thước dữ liệu và yêu cầu tốc độ,
bạn có thể triển khai matcher bằng phép tính cosine đơn thuần hoặc dùng
thư viện tìm kiếm gần đúng (FAISS, Annoy…).
"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

class BaseMatcher(ABC):
    """Abstract base class for embedding matchers."""

    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold: Ngưỡng similarity. Nếu điểm lớn hơn hoặc bằng ngưỡng,
                       coi là match; ngược lại trả về 'unknown'.
        """
        self.threshold = threshold

    @abstractmethod
    def match(self, query: np.ndarray) -> Tuple[str, float]:
        """
        Tìm nhãn phù hợp nhất cho embedding query.

        Args:
            query: Vector embedding 1D.

        Returns:
            Tuple (label, score). Nếu không vượt ngưỡng, label có thể là 'unknown'.
        """
        raise NotImplementedError

    def __call__(self, query: np.ndarray) -> Tuple[str, float]:
        """Cho phép gọi matcher như hàm."""
        return self.match(query)