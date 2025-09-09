import numpy as np
from typing import List, Tuple
from argparse import Namespace
from ultralytics.trackers.byte_tracker import BYTETracker
from .base import BaseTracker

class _BoxesLite:
    """Tối thiểu hóa cấu trúc giống ultralytics.engine.results.Boxes"""
    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy = np.asarray(xyxy, dtype=np.float32)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls  = np.asarray(cls,  dtype=np.float32).reshape(-1)

    @property
    def xywh(self):
        if len(self.xyxy) == 0:
            return self.xyxy.reshape(0, 4)
        x1, y1, x2, y2 = self.xyxy.T
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        return np.stack([cx, cy, w, h], axis=1)

    def __len__(self):
        return self.xyxy.shape[0]

    def __getitem__(self, idx):
        return _BoxesLite(self.xyxy[idx], self.conf[idx], self.cls[idx])

class ByteTrackTracker(BaseTracker):
    """
    Wrapper cho ByteTrack.
    """

    def __init__(self, frame_rate: int = 30, track_thresh: float = 0.5,
                 match_thresh: float = 0.8, min_box_area: float = 10.0, use_cuda: bool = False):

        device = "cpu"
        if use_cuda:
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda:0"
            except Exception:
                device = "cpu"

        track_buffer = max(int(round(frame_rate * 0.5)), 1)  # ~0.5s

        # Bổ sung các tham số mà BYTETracker.update cần dùng
        args = Namespace(
            tracker_type="bytetrack",
            track_high_thresh=float(track_thresh),
            track_low_thresh=0.1,
            new_track_thresh=0.6,
            match_thresh=float(match_thresh),
            track_buffer=int(track_buffer),
            fuse_score=True,
            min_box_area=float(min_box_area),  # không dùng nhiều trong bản Ultralytics, nhưng giữ cho tương thích
            mot20=False,
            device=device,
            use_cuda=(device != "cpu"),
        )

        self.tracker = BYTETracker(args, frame_rate=int(frame_rate))
        self.frame_rate = int(frame_rate)

    def update(self,
               detections: List[Tuple[int, int, int, int, float]],
               frame_shape: Tuple[int, int]) -> List[Tuple[int, int, int, int, int]]:
        """
        Update tracker và trả về danh sách (x, y, w, h, id).
        Input detections: (x, y, w, h, score) theo tọa độ góc trên-trái + kích thước.
        """

        if len(detections) == 0:
            det_view = _BoxesLite(np.zeros((0, 4), np.float32),
                                  np.zeros((0,), np.float32),
                                  np.zeros((0,), np.float32))
        else:
            xyxy = []
            conf = []
            cls  = []
            for (x, y, w, h, score) in detections:
                x1, y1 = x, y
                x2, y2 = x + w, y + h
                xyxy.append([x1, y1, x2, y2])
                conf.append(score)
                cls.append(0.0)  # nếu đơn lớp, gán 0; thay đổi theo use case của bạn
            det_view = _BoxesLite(np.asarray(xyxy, np.float32),
                                  np.asarray(conf, np.float32),
                                  np.asarray(cls,  np.float32))

        # có thể truyền ảnh gốc nếu bạn cần GMC hoặc đặc trưng ReID; ở đây để None
        tracks = self.tracker.update(det_view, img=None, feats=None)  # trả về np.ndarray
        results = []
        for t in tracks:
            # Định dạng: [x1, y1, x2, y2, conf, cls, track_id, idx]
            x1, y1, x2, y2 = t[:4]
            track_id = int(t[-2])      # cột kế cuối
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            results.append((x, y, w, h, track_id))
        return results
