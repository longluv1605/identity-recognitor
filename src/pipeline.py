"""
Realtime face recognition pipeline orchestrator.

Lớp FaceRecognitionPipeline khởi tạo các thành phần từ cấu hình YAML
và cung cấp phương thức process_frame() để xử lý từng khung hình.
"""

from typing import List, Tuple
import yaml
import numpy as np

from .detectors.yolov8_detector import YoloDetector
from .aligners.mediapipe_aligner import MediaPipeAligner
from .embedders.simple_embedder import SimpleEmbedder
from .matchers.simple_matcher import SimpleMatcher
from .trackers.deepsort_tracker import DeepSortTracker
from .database import EmbeddingDatabase

class FaceRecognitionPipeline:
    def __init__(self, config_path: str, db_path: str = None):
        """
        Khởi tạo các module theo cấu hình.

        Args:
            config_path: đường dẫn file YAML cấu hình.
            db_path: đường dẫn file pickle database embeddings.
        """
        try:
            with open(config_path, "r", encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
        except Exception as e:
            raise e

        # Chọn detector
        det_cfg = cfg.get("detector", {})
        det_name = det_cfg.get("name", "haar")
        if det_name == "yolov8":
            self.detector = YoloDetector(det_cfg.get("model_path", "yolov8n.pt"))
        else:
            raise ValueError(f"Unsupported detector: {det_name}")
        
        # Chọn aligner
        align_cfg = cfg.get("aligner", {})
        align_type = align_cfg.get("type", "mediapipe")
        if align_type == "mediapipe":
            self.aligner = MediaPipeAligner(
                output_size=tuple(align_cfg.get("output_size", [112, 112])),
                detection_confidence=align_cfg.get("detection_confidence", 0.5),
            )
        else:
            raise ValueError(f"Unsupported aligner: {align_type}")

        # Chọn embedder
        emb_cfg = cfg.get("embedding", {})
        emb_method = emb_cfg.get("method", "simple")
        if emb_method == "simple":
            self.embedder = SimpleEmbedder(
                output_dim=emb_cfg.get("output_dim"), size=emb_cfg.get("input_size", 32)
            )
        else:
            raise ValueError(f"Unsupported embedding method: {emb_method}")

        # Tải database
        self.db = EmbeddingDatabase(db_path) if db_path else None

        # Chọn matcher
        match_cfg = cfg.get("matcher", {})
        matcher_type = match_cfg.get("type", "simple")
        threshold = match_cfg.get("threshold", 0.5)
        if matcher_type == "simple":
            self.matcher = SimpleMatcher(
                embeddings=self.db.get_all() if self.db else {}, threshold=threshold
            )
        else:
            raise ValueError(f"Unsupported matcher: {matcher_type}")

        # Chọn tracker
        track_cfg = cfg.get("tracker", {})
        tracker_type = track_cfg.get("type", "simple")
        if tracker_type == "deepsort":
            self.tracker = DeepSortTracker(
                max_age=track_cfg.get("max_age", 5),
                n_init=track_cfg.get("n_init", 3),
                max_iou_distance=track_cfg.get("max_iou_distance", 0.7),
                embedder_gpu=track_cfg.get("embedder_gpu", False),
                half=track_cfg.get("half", False),
                bgr=track_cfg.get("bgr", True),
            )
        else:
            raise ValueError(f"Unsupported tracker: {tracker_type}")

    def process_frame(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, str, float, int]]:
        """
        Xử lý một khung hình: detect -> align -> embed -> match -> track.

        Returns:
            Danh sách tuple (x, y, w, h, label, score, track_id).
        """
        # 1. Phát hiện khuôn mặt
        bboxes = self.detector.detect(frame)
        detections_with_score = []

        # Nếu detector của bạn không trả điểm confidence, đặt mặc định
        for bbox in bboxes:
            # bbox = (x, y, w, h)
            detections_with_score.append((*bbox, 1.0))

        # 2. Tracking: cần track trước khi nhận dạng nếu dùng Track
        if isinstance(self.tracker, DeepSortTracker):
            tracks = self.tracker.update(detections_with_score, frame)
            # Deepsort đã gán id; ta tạm giữ để xử lý từng track
            results = []
            for x, y, w, h, tid in tracks:
                face_aligned = self.aligner(frame, (x, y, w, h))
                embedding = self.embedder.embed(face_aligned)
                label, score = self.matcher.match(embedding)
                results.append((x, y, w, h, label, score, tid))
            return results
        else:
            # 3. Nếu tracker đơn giản: xử lý detection rồi gán id sau
            results = []
            for (x, y, w, h, score) in detections_with_score:
                face_aligned = self.aligner(frame, (x, y, w, h))
                embedding = self.embedder.embed(face_aligned)
                label, sim = self.matcher.match(embedding)
                results.append((x, y, w, h, label, sim, -1))
            # Gán ID tăng dần
            if self.tracker:
                tracked = self.tracker.assign_ids([(x, y, w, h) for (x, y, w, h, _, _, _) in results])
                # Kết hợp id vào kết quả
                final_results = []
                for ((x, y, w, h, label, sim, _), (_, _, _, _, tid)) in zip(results, tracked):
                    final_results.append((x, y, w, h, label, sim, tid))
                return final_results
            return results