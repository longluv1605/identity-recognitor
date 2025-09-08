"""
Simple Face Aligner - không dùng landmarks, chỉ crop và resize
"""
from typing import Tuple
import cv2
import numpy as np
from .base import BaseAligner

class SimpleAligner(BaseAligner):
    """
    Simple aligner chỉ crop face từ bbox và resize về target size.
    Không dùng landmarks nên không có alignment thực sự, 
    nhưng ổn định và consistent hơn MediaPipe khi landmarks detection kém.
    """

    def __init__(self, output_size: Tuple[int, int] = (112, 112), 
                 padding_ratio: float = 0.2):
        """
        Args:
            output_size: Kích thước output (width, height)
            padding_ratio: Tỷ lệ padding xung quanh face bbox
        """
        self.output_size = output_size
        self.padding_ratio = padding_ratio

    def align(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Crop face với padding và resize về output_size
        
        Args:
            frame: ảnh BGR gốc
            bbox: bounding box (x, y, w, h)
            
        Returns:
            Ảnh face đã crop và resize
        """
        x, y, w, h = bbox
        h_frame, w_frame = frame.shape[:2]
        
        # Thêm padding
        pad_w = int(w * self.padding_ratio)
        pad_h = int(h * self.padding_ratio)
        
        # Tính toán crop area với padding
        x_start = max(0, x - pad_w)
        y_start = max(0, y - pad_h)
        x_end = min(w_frame, x + w + pad_w)
        y_end = min(h_frame, y + h + pad_h)
        
        # Crop face
        face_crop = frame[y_start:y_end, x_start:x_end]
        
        if face_crop.size == 0:
            return np.zeros((*self.output_size, 3), dtype=np.uint8)
        
        # Resize về target size
        aligned = cv2.resize(face_crop, self.output_size, interpolation=cv2.INTER_LINEAR)
        
        return aligned
