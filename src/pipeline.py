"""
Realtime face recognition pipeline orchestrator.

Lớp FaceRecognitionPipeline khởi tạo các thành phần từ cấu hình YAML
và cung cấp phương thức process_frame() để xử lý từng khung hình.
"""

from typing import List, Tuple
import yaml
import numpy as np

from src.detectors import YoloDetector
from src.aligners import MediaPipeAligner, SimpleAligner
from src.embedders import SimpleEmbedder, ArcFaceEmbedder, DeepFaceEmbedder
from src.matchers import SimpleMatcher
from src.trackers import DeepSortTracker, ByteTrackTracker
from src.database import EmbeddingDatabase

class FaceRecognitionPipeline:
    def __init__(self, config_path: str):
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
            raise ValueError(f'Not valid config path: {e}')

        # Chọn detector
        det_cfg = cfg.get("detector", {})
        self._create_detector(det_cfg)
        
        # Chọn aligner
        align_cfg = cfg.get("aligner", {})
        self._create_aligner(align_cfg)

        # Chọn embedder
        emb_cfg = cfg.get("embedding", {})
        self._create_embedder(emb_cfg)

        # Tải database
        db_cfg = cfg.get("database", {})
        self._load_database(db_cfg)

        # Chọn matcher
        match_cfg = cfg.get("matcher", {})
        self._create_matcher(match_cfg)

        # Chọn tracker
        track_cfg = cfg.get("tracker", {})
        self._create_tracker(track_cfg)

    def process_frame(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, str, float, int]]:
        """
        Xử lý một khung hình: detect -> align -> embed -> match -> track.

        Returns:
            Danh sách tuple (x, y, w, h, label, score, track_id).
        """
        # detect → chuẩn hoá (x, y, w, h, score)
        bboxes = self.detector.detect(frame) or []
        detections = [(*b[:4], (b[4] if len(b) >= 5 else 1.0)) for b in bboxes]

        def recognize(x: int, y: int, w: int, h: int) -> Tuple[str, float]:
            face = self.aligner.align(frame, (x, y, w, h))
            emb = self.embedder.embed(face)
            return self.matcher.match(emb)

        # track nếu có tracker “thực”
        tracks = None
        if isinstance(self.tracker, DeepSortTracker):
            tracks = self.tracker.update(detections, frame)
        elif isinstance(self.tracker, ByteTrackTracker):
            tracks = self.tracker.update(detections, frame.shape[:2])

        if tracks:
            return [(x, y, w, h, *recognize(x, y, w, h), tid) for x, y, w, h, tid in tracks]

        # không có tracker/hoặc tracker đơn giản → nhận dạng trước
        results = [(x, y, w, h, *recognize(x, y, w, h), -1) for (x, y, w, h, _) in detections]

        # gán ID nếu có assign_ids
        if getattr(self.tracker, "assign_ids", None):
            ids = self.tracker.assign_ids([(x, y, w, h) for (x, y, w, h, _, _, _) in results])
            return [(x, y, w, h, label, score, tid)
                    for (x, y, w, h, label, score, _), (_, _, _, _, tid) in zip(results, ids)]

        return results
    
    def _load_database(self, db_cfg):
        self.db = EmbeddingDatabase(db_cfg.get("output")) if db_cfg.get("output") else None
    
    def _create_detector(self, det_cfg):
        det_name = det_cfg.get("name", "yolov8")
        if det_name == "yolov8":
            self.detector = YoloDetector(det_cfg.get("model_path", "yolov8n.pt"))
        else:
            raise ValueError(f"Unsupported detector: {det_name}")
    
    def _create_aligner(self, align_cfg):
        align_type = align_cfg.get("type", "mediapipe")
        if align_type == "mediapipe":
            self.aligner = MediaPipeAligner(
                output_size=tuple(align_cfg.get("output_size", [112, 112])),
                detection_confidence=align_cfg.get("detection_confidence", 0.5),
            )
        elif align_type == "simple":
            self.aligner = SimpleAligner(
                output_size=tuple(align_cfg.get("output_size", [112, 112]))
            )
        else:
            raise ValueError(f"Unsupported aligner: {align_type}")
    
    def _create_embedder(self, emb_cfg):
        emb_method = emb_cfg.get("method", "simple")
        if emb_method == "simple":
            self.embedder = SimpleEmbedder(
                output_dim=emb_cfg.get("output_dim"), size=emb_cfg.get("input_size", 32)
            )
        elif emb_method == "arcface":
            self.embedder = ArcFaceEmbedder(
                model_path=emb_cfg.get("model_path"), device=emb_cfg.get("device", "cpu")                     
            )
        elif emb_method == "deepface":
            self.embedder = DeepFaceEmbedder(
                model_name=emb_cfg.get("model_name"),
                enforce_detection=emb_cfg.get("enforce_detection", False)
            )
        else:
            raise ValueError(f"Unsupported embedding method: {emb_method}")
        
    def _create_matcher(self, match_cfg):
        matcher_type = match_cfg.get("type", "simple")
        threshold = match_cfg.get("threshold", 0.5)
        use_cosine = match_cfg.get("use_cosine", True)  # mặc định dùng cosine
        if matcher_type == "simple":
            self.matcher = SimpleMatcher(
                embeddings=self.db.get_all() if self.db else {}, 
                threshold=threshold,
                use_cosine=use_cosine
            )
        else:
            raise ValueError(f"Unsupported matcher: {matcher_type}")
    
    def _create_tracker(self, track_cfg):
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
        elif tracker_type == "bytetrack":
            self.tracker = ByteTrackTracker(
                frame_rate=track_cfg.get("frame_rate", 30),
                track_thresh=track_cfg.get("track_thresh", 0.5),
                match_thresh=track_cfg.get("match_thresh", 0.8),
                min_box_area=track_cfg.get("min_box_area", 10),
                use_cuda=track_cfg.get("use_cuda", False),
            )
        else:
            raise ValueError(f"Unsupported tracker: {tracker_type}")