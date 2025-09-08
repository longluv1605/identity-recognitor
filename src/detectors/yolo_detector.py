import numpy as np
from typing import List, Tuple
from ultralytics import YOLO
from .base import BaseDetector


class YoloDetector(BaseDetector):
    """YOLO detector using ultralytics."""

    def __init__(self, model: str = "yolov8n.pt", device: str = "cpu"):
        """
        Args:
            model: Đường dẫn tới file .pt của mô hình YOLOv8 được huấn luyện cho
                   phát hiện khuôn mặt (hoặc tên model có sẵn như 'yolov8n.pt').
            device: Thiết bị chạy mô hình ('cpu' hoặc 'cuda').
        """
        super().__init__()
        # Tải mô hình YOLO
        self.model = YOLO(model)
        self.device = device
        # Đảm bảo mô hình chạy trên thiết bị mong muốn
        self.model.to(device)

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Nhận một khung hình BGR và trả về danh sách các bounding box (x, y, w, h).

        Ultralytics YOLO trả về tọa độ (x1, y1, x2, y2); chúng ta chuyển đổi
        sang (x, y, w, h) để tương thích với pipeline.
        """
        # YOLO mong đầu vào là RGB
        img = frame[:, :, ::-1]
        results = self.model(img, verbose=False)
        boxes = []
        if len(results) > 0:
            # results[0] tương ứng với lô ảnh đầu tiên (chỉ một ảnh)
            result = results[0]
            if hasattr(result, "boxes") and len(result.boxes) > 0:
                # Lấy tọa độ bounding box ở dạng xyxy (tensor trên GPU)
                xyxy = result.boxes.xyxy.cpu().numpy()
                for (x1, y1, x2, y2) in xyxy:
                    x, y = int(x1), int(y1)
                    w, h = int(x2 - x1), int(y2 - y1)
                    boxes.append((x, y, w, h))
        return boxes