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
import yaml
from tqdm import tqdm

from src.detectors import YoloDetector
from src.aligners import SimpleAligner
from src.embedders import ArcFaceEmbedder, DeepFaceEmbedder
from src.database import save_database


def build_database(config_path: str) -> None:
    try:
        with open(config_path, "r", encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        raise e
    
    det_cfg = cfg.get('detector', {})
    align_cfg = cfg.get('aligner', {})
    emb_cfg = cfg.get('embedding', {})
    db_cfg = cfg.get('database', {})
    
    if det_cfg['name'] == 'yolov8':    
        detector = YoloDetector(model=cfg['detector']['model_path'])
    else:
        raise ValueError("Unsupported detector...")
        
    if align_cfg['type'] == 'simple':
        # aligner = MediaPipeAligner(output_size=align_cfg['output_size'])
        aligner = SimpleAligner(output_size=align_cfg['output_size'])
    else:
        raise ValueError("Unsupported aligner...")
        
    if emb_cfg['method'] == 'deepface':
        embedder = DeepFaceEmbedder(model_name=emb_cfg['model_name'])
    elif emb_cfg['method'] == 'arcface':
        embedder = ArcFaceEmbedder(model_path=emb_cfg['model_path'])
    else:
        raise ValueError(f"Unsupported embedder [{emb_cfg['method']}]...")

    db = {}
    people = [d for d in os.listdir(db_cfg['input']) if os.path.isdir(os.path.join(db_cfg['input'], d))]
    for person in people:
        person_dir = os.path.join(db_cfg['input'], person)
        save_dir = os.path.join(db_cfg['processed'], person)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        embeddings = []
        loop = tqdm(os.listdir(person_dir), desc=f"Processing [{person}]")
        for img_name in loop:
            img_path = os.path.join(person_dir, img_name)
            save_path = os.path.join(save_dir, img_name)
            
            img = cv2.imread(img_path)
            if img is None:
                continue
            # detect faces
            bboxes = detector.detect(img)
            for (x, y, w, h, _, _) in bboxes:
                # align and embed
                face = aligner.align(img, (x, y, w, h))
                vec = embedder.embed(face)
                embeddings.append(vec)
                cv2.imwrite(save_path, face)
        if embeddings:
            # Lưu tất cả embeddings thay vì chỉ mean
            # Format: list of embeddings cho mỗi người
            db[person] = np.array(embeddings)
    save_database(db, db_cfg['output'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create face embedding database")
    parser.add_argument("--config", default='config/config.yaml', help="Path to config file")
    args = parser.parse_args()
    build_database(args.config)
    print('=> Database built successfully...')