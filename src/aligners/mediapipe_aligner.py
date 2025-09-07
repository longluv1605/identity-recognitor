"""
Face alignment class using MediaPipe Face Mesh.

Lớp này nhận một bounding box và trích xuất landmark để xoay và scale khuôn mặt
sao cho mắt nằm ngang, phù hợp với mô hình embedding như ArcFace.
"""

from typing import Tuple
import cv2
import numpy as np
import mediapipe as mp
from .base import BaseAligner

class MediaPipeAligner(BaseAligner):
    # Cố định một số ID landmark cho mắt và mũi
    LEFT_EYE_IDX = 33   # khóe mắt trái ngoài
    RIGHT_EYE_IDX = 263 # khóe mắt phải ngoài
    NOSE_TIP_IDX = 1    # đầu mũi

    def __init__(self,
                 output_size: Tuple[int, int] = (112, 112),
                 detection_confidence: float = 0.5):
        """
        Args:
            output_size: Kích thước ảnh output (width, height) sau căn chỉnh.
            detection_confidence: ngưỡng confidence khi tìm landmark.
        """
        self.output_size = output_size
        # Khởi tạo FaceMesh một lần với cấu hình tĩnh
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=detection_confidence,
        )

        # Template điểm đích cho mắt trái, mắt phải, mũi; phù hợp với ArcFace (112×112)
        self._template = np.float32([
            [30.2946, 51.6963],   # vị trí mắt trái trong ảnh chuẩn
            [81.6958, 51.5014],   # mắt phải
            [56.0,    71.7366],   # mũi
        ])

        # Nếu kích thước khác 112x112, cần scale template tương ứng
        if output_size != (112, 112):
            scale_x = output_size[0] / 112.0
            scale_y = output_size[1] / 112.0
            self._template = self._template * np.array([scale_x, scale_y], dtype=np.float32)

    def align(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Căn chỉnh khuôn mặt theo bounding box.

        Args:
            frame: ảnh BGR gốc.
            bbox: bounding box (x, y, w, h).

        Returns:
            Ảnh khuôn mặt đã căn chỉnh và resize về output_size.
        """
        x, y, w, h = bbox
        h_frame, w_frame = frame.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w_frame, x + w), min(h_frame, y + h)

        # Cắt ROI khuôn mặt
        face_roi = frame[y0:y1, x0:x1]
        if face_roi.size == 0:
            return np.zeros((*self.output_size, 3), dtype=np.uint8)

        # Chuyển sang RGB cho MediaPipe
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        results = self._mesh.process(face_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            h_roi, w_roi = face_roi.shape[:2]

            # Lấy tọa độ 3 điểm nguồn
            left_eye = np.array(
                [landmarks[self.LEFT_EYE_IDX].x * w_roi,
                 landmarks[self.LEFT_EYE_IDX].y * h_roi],
                dtype=np.float32
            )
            right_eye = np.array(
                [landmarks[self.RIGHT_EYE_IDX].x * w_roi,
                 landmarks[self.RIGHT_EYE_IDX].y * h_roi],
                dtype=np.float32
            )
            nose_tip = np.array(
                [landmarks[self.NOSE_TIP_IDX].x * w_roi,
                 landmarks[self.NOSE_TIP_IDX].y * h_roi],
                dtype=np.float32
            )
            src_pts = np.stack([left_eye, right_eye, nose_tip], axis=0)

            # Tính ma trận affine và warp ảnh
            M = cv2.getAffineTransform(src_pts, self._template)
            aligned = cv2.warpAffine(
                face_roi,
                M,
                self.output_size,
                flags=cv2.INTER_LINEAR,
                borderValue=0,
            )
            return aligned

        # Trường hợp không có landmark: trả về vùng crop resize
        return cv2.resize(face_roi, self.output_size)