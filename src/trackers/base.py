from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

class BaseTracker(ABC):
    """Abstract base class for multi-object trackers."""

    @abstractmethod
    def update(
        self,
        detections: List[Tuple[int, int, int, int, float]],
        frame_shape: Tuple[int, int]
    ) -> List[Tuple[int, int, int, int, int]]:
        """
        Cập nhật tracker với danh sách detection mới.

        Args:
            detections: danh sách tuple (x, y, w, h, score) của mỗi đối tượng
                        được detector trả về.
            frame_shape: chiều cao và chiều rộng khung hình (h, w) – cần cho ByteTrack.

        Returns:
            Danh sách tuple (x, y, w, h, track_id) sau khi gán ID.
        """
        raise NotImplementedError
    
    def __call__(
        self,
        detections: List[Tuple[int, int, int, int, float]],
        frame_shape: Tuple[int, int]
    ) -> List[Tuple[int, int, int, int, int]]:
        """Cho phép gọi matcher như hàm."""
        return self.update(detections, frame_shape)