"""
Build a face embedding database from a folder of labelled images.

Usage:
    python scripts/build_db.py --input data/raw --output data/embeddings/db.pkl --model models/detection/yolo/yolov8n-face.pt

Thư mục input phải có cấu trúc:
    data/raw/
        long/
            *.jpg
    ...

Mỗi thư mục con là tên của một người (nhãn) và chứa các ảnh khuôn mặt của họ.
"""

import os
import sys

##############
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
##############

import argparse
import cv2
import numpy as np
from tqdm import tqdm

from src.detectors.yolo_detector import YoloDetector
from src.aligners.mediapipe_aligner import MediaPipeAligner
from src.embedders.arcface_embedder import ArcFaceEmbedder
from src.database import save_database


def build_database(input_dir: str, output_path: str, detect_model: str, embed_model: str) -> None:
    detector = YoloDetector(model=detect_model)
    aligner = MediaPipeAligner(output_size=(112, 112), detection_confidence=0.5)
    embedder = ArcFaceEmbedder(model_path=embed_model)

    db = {}
    people = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    # people = tqdm(people, desc='Building database: ')
    for person in people:
        person_dir = os.path.join(input_dir, person)
        embeddings = []
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            # detect faces
            bboxes = detector.detect(img)
            for (x, y, w, h) in bboxes:
                # align and embed
                face = aligner(img, (x, y, w, h))
                vec = embedder.embed(face)
                embeddings.append(vec)
        if embeddings:
            db[person] = np.mean(embeddings, axis=0)
    save_database(db, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create face embedding database")
    parser.add_argument("--input", required=True, help="Path to labelled face image directory")
    parser.add_argument("--output", required=True, help="Path to output pickle file")
    parser.add_argument(
        "--detect_model", default="models/detection/yolo/yolov8n-face.pt", help="YOLOv8 face detection model (.pt file)"
    )
    parser.add_argument(
        "--embed_model", default="models/embedding/arcface/arcfaceresnet100-11-int8.onnx", help="YOLOv8 face detection model (.pt file)"
    )
    args = parser.parse_args()
    build_database(args.input, args.output, args.detect_model, args.embed_model)
    print('=> Database built successfully...')