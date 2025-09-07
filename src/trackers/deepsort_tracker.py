from typing import List, Tuple
import numpy as np
from .base import BaseTracker

class DeepSortTracker(BaseTracker):
    """
    Multi-object tracker using the deep-sort-realtime library.
    Requires: pip install deep-sort-realtime.
    """
    def __init__(self, max_age=5, n_init=3, max_iou_distance=0.7,
                 embedder_gpu=False, half=False, bgr=True):
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
        except ImportError as e:
            raise ImportError("deep-sort-realtime is required for DeepSortTracker. "
                              "Install it with 'pip install deep-sort-realtime'.") from e
        self._tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max_iou_distance,
            embedder_gpu=embedder_gpu,
            half=half,
            bgr=bgr,
        )

    def update(self,
               detections: List[Tuple[int, int, int, int, float]],
               frame: np.ndarray) -> List[Tuple[int, int, int, int, int]]:
        # Chuyển detections về ([x, y, w, h], score, class_id)
        bbs = [([x, y, w, h], float(score), 0) for (x, y, w, h, score) in detections]
        tracks = self._tracker.update_tracks(bbs, frame=frame)
        results: List[Tuple[int, int, int, int, int]] = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, r, b = track.to_ltrb()
            x, y, w, h = int(l), int(t), int(r - l), int(b - t)
            results.append((x, y, w, h, track_id))
        return results