"""
ArcFace embedding extractor.

Lớp này bao bọc một mô hình ArcFace đã huấn luyện (định dạng ONNX) và
trích xuất vector đặc trưng 512 chiều. Bạn cần cài đặt gói onnxruntime
hoặc onnxruntime‑gpu và cung cấp đường dẫn tới file .onnx.
"""

from typing import Optional
import numpy as np
import cv2

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from .simple_embedder import SimpleEmbedder


class ArcFaceEmbedder(SimpleEmbedder):
    """
    Embedder sử dụng mô hình ArcFace. Kế thừa SimpleEmbedder để giữ tham số
    size và output_dim, nhưng override phương thức embed().

    Parameters
    ----------
    model_path : str
        Đường dẫn tới file ArcFace ONNX (bắt buộc).
    device : str, optional
        Thiết bị thực thi cho ONNX Runtime ('cpu' hoặc 'cuda').
        Mặc định là 'cpu'; để dùng GPU cần cài onnxruntime‑gpu.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        # Gọi parent để đặt output_dim=512, input size=112
        super().__init__(output_dim=512, size=112)
        if ort is None:
            raise ImportError(
                "ArcFaceEmbedder requires the onnxruntime package. "
                "Install it with 'pip install onnxruntime' or 'pip install onnxruntime-gpu'."
            )
        if not model_path:
            raise ValueError("An ArcFace ONNX model path must be provided")

        # Chọn provider tuỳ theo device
        if device.lower() == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        # Khởi tạo session ONNX
        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
        except Exception as e:
            raise RuntimeError(f"Failed to load ArcFace model from '{model_path}': {e}")

    def embed(self, face: np.ndarray) -> np.ndarray:
        """Trả về vector 512 chiều đã chuẩn hoá cho ảnh khuôn mặt."""
        if face.size == 0:
            return np.zeros(self.output_dim, dtype=np.float32)
        # Resize về 112x112
        img = cv2.resize(face, (self.size, self.size))
        # Chuyển BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Đổi sang float32 và scale về [0,1]
        img = img.astype(np.float32) / 255.0
        # Chuẩn hoá [-1,1]
        img = (img - 0.5) / 0.5
        # Chuyển shape (H,W,C) -> (C,H,W)
        img = np.transpose(img, (2, 0, 1))
        # Thêm dimension batch
        img = np.expand_dims(img, axis=0)

        # Tên input lấy từ mô hình
        ort_inputs = {self.session.get_inputs()[0].name: img}
        embeddings = self.session.run(None, ort_inputs)[0]
        vec = embeddings[0].astype(np.float32)
        # Chuẩn hoá vector về độ dài 1
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec
