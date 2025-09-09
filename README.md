# Realtime Face Recognition System

Hệ thống nhận diện khuôn mặt thời gian thực sử dụng Deep Learning với YOLOv8 face detection, DeepFace embeddings và ByteTrack tracking.

## Tính năng chính

- **Face Detection**: YOLOv8n-face model cho detection nhanh và chính xác
- **Face Embedding**: Hỗ trợ nhiều models (DeepFace, ArcFace, Facenet512, VGG-Face...)
- **Face Recognition**: So khớp với database embeddings đa dạng
- **Real-time Tracking**: ByteTrack/DeepSort để track faces qua các frame
- **FPS Display**: Monitor hiệu suất real-time
- **Configurable**: Cấu hình linh hoạt qua YAML
- **Multiple Embeddings**: Hỗ trợ nhiều embeddings per person để tăng độ chính xác

## Kiến trúc hệ thống

```plain
Input Frame → Face Detection → Face Alignment → Embedding Extraction → Matching → Tracking   →  Output
     ↓              ↓               ↓                 ↓                  ↓              ↓         ↓
  Webcam         YOLOv8 (Face)   MediaPipe       DeepFace/ArcFace    Database       ByteTrack/  Display
                                                                     Matching       DeepSORT     + FPS
```

## Cấu trúc thư mục

```plain
realtime-face-recognition/
├── config/
│   └── config.yaml              # Cấu hình chính
├── data/
│   ├── raw/                     # Ảnh gốc (trong này sẽ có các thư mục con (tên người) chứa ảnh của mỗi người)
│   ├── processed/               # Ảnh đã xử lý
│   ├── embeddings/
│   │   └── db.pkl              # Database embeddings
│   └── test/                   # Ảnh test
├── models/
│   ├── detection/
│   │   └── yolo/
│   │       └── yolov8n-face.pt # YOLOv8 face detection model
│   ├── embedding/
│   │   └── arcface/            # ArcFace models
│   └── tracking/               # Tracking models
├── src/
│   ├── detectors/              # Face detection modules
│   ├── aligners/               # Face alignment modules
│   ├── embedders/              # Embedding extraction modules
│   ├── matchers/               # Face matching modules
│   ├── trackers/               # Face tracking modules
│   ├── utils/                  # Utilities
│   ├── pipeline.py             # Main pipeline
│   ├── database.py             # Database management
│   └── main.py                 # Entry point
├── scripts/
│   ├── build_db.py             # Xây dựng database embeddings
│   ├── evaluate.py             # Đánh giá model
│   └── export_onnx.py          # Export models
└── tests/                      # Unit tests
```

## Cài đặt

### 1. Tạo môi trường Conda

-> **Yêu cầu cần có Conda trên hệ thống**

```bash
# Tạo environment từ file YAML
conda env create -f environment.yml

# Kích hoạt environment
conda activate realtime-face-recognition
```

### 3. Download models

```bash
# Hoặc tải thủ công tại: https://github.com/akanametov/yolov8-face
```

## Chuẩn bị dữ liệu

### 1. Cấu trúc dữ liệu

```plain
data/raw/
├── person1/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── person2/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── ...
```

### 2. Xây dựng database

```bash
# Tạo database embeddings từ ảnh raw
python scripts/build_db.py [--config config/config.yaml]
```

Database sẽ được lưu tại `data/embeddings/db.pkl` với format:

- Hỗ trợ **nhiều embeddings per person** để tăng độ robust
- Mỗi person có array của embeddings thay vì 1 embedding trung bình

## Sử dụng

### 1. Chạy real-time recognition

```bash
# Sử dụng webcam mặc định (camera 0)
python src/main.py [--config config/config.yaml]
```

### 2. Các phím tắt

- **'q'**: Thoát ứng dụng
- **FPS display**: Hiển thị tự động ở góc trái màn hình

### 3. Output format

```plain
Bounding box: Màu xanh lá quanh khuôn mặt
Label: Tên person + confidence score + Track ID
FPS: Hiển thị ở góc trái trên
```

## Models hỗ trợ

### Face Detection

- **YOLOv8n-face**: Nhanh, nhẹ, phù hợp real-time

### Face Embedding

- **DeepFace**: Facenet512, ArcFace, VGG-Face, OpenFace, DeepID, Dlib, SFace
- **ArcFace**: ONNX models cho performance cao
- **Simple**: CNN đơn giản cho test

### Face Alignment

- **MediaPipe**: Sử dụng landmarks cho alignment chính xác
- **SimpleResize**: Chỉ crop và resize, ổn định hơn

### Face Tracking

- **ByteTrack**: Nhanh, chính xác, ít resource
- **DeepSort**: Sử dụng appearance features

## Đóng góp

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

## 📄 License

GNU License. Xem `LICENSE` để biết thêm thông tin.

## Liên hệ

- **Email**: [longluv1605@gmail.com]
- **GitHub**: [longluv1605](https://github.com/longluv1605)
- **Project Link**: [https://github.com/longluv1605/identity-recognitor](https://github.com/longluv1605/identity-recognitor)

## 🙏 Acknowledgments

- [YOLOv8](https://github.com/ultralytics/ultralytics) - Face detection model
- [DeepFace](https://github.com/serengil/deepface) - Face recognition framework
- [ByteTrack](https://github.com/ifzhang/ByteTrack) - Multi-object tracking
- [MediaPipe](https://mediapipe.dev/) - Face landmarks detection

---

⭐ **Nếu project này hữu ích, hãy cho một star!** ⭐
