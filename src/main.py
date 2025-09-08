"""
Entry point for realtime face recognition demo.

Khởi tạo pipeline từ file cấu hình và (tuỳ chọn) file database,
rồi đọc khung hình từ webcam, xử lý và hiển thị kết quả.
"""

import os
import sys

##############
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
##############

import argparse
import cv2
import time

from src.pipeline import FaceRecognitionPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Realtime Face Recognition Demo")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml"),
        help="Đường dẫn file cấu hình YAML",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "data", "embeddings", "db.pkl"),
        help="Đường dẫn database embeddings (pickle). Nếu không có, hệ thống sẽ luôn trả 'unknown'",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="ID của webcam (0, 1, ...). Nếu không chỉ định sẽ lấy từ config.yaml",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(args)
    pipeline = FaceRecognitionPipeline(config_path=args.config, db_path=args.db)

    # Lấy camera ID từ arg hoặc config
    camera_id = args.camera
    if camera_id is None:
        import yaml
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        camera_id = cfg.get("camera", {}).get("device_id", 0)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Không thể mở webcam {camera_id}")

    # Khởi tạo biến để tính FPS
    prev_time = time.time()
    fps_display = 0.0

    print("Nhấn 'q' để thoát.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không lấy được khung hình.")
            break

        results = pipeline.process_frame(frame)
        print(results)
        
        # Vẽ bounding box và nhãn
        for x, y, w, h, label, score, track_id in results:
            # rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # label text
            text = f"{label} {score:.2f}"
            text += f" ID:{track_id}"
            cv2.putText(
                frame,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
        
        # Tính toán FPS chính xác cho từng frame
        current_time = time.time()
        time_diff = current_time - prev_time
        if time_diff > 0:
            fps_display = 1.0 / time_diff
        prev_time = current_time
        
        # Hiển thị FPS lên góc trái màn hình
        fps_text = f"FPS: {fps_display:.1f}"
        cv2.putText(
            frame,
            fps_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),  # Màu cyan
            2
        )
        
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    
#########################################
if __name__ == '__main__':
    main()