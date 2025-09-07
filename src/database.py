"""
Embedding database utilities.

Cung cấp lớp EmbeddingDatabase để quản lý việc nạp/lưu embeddings
(từ file pickle) và cập nhật trong quá trình chạy.
"""

import os
import pickle
from typing import Dict
import numpy as np


class EmbeddingDatabase:
    def __init__(self, path: str):
        """
        Args:
            path: đường dẫn tới file pickle lưu database.
        """
        self.path = path
        self.embeddings: Dict[str, np.ndarray] = self._load()

    def _load(self) -> Dict[str, np.ndarray]:
        if not os.path.exists(self.path):
            return {}
        with open(self.path, "rb") as f:
            data = pickle.load(f)
        # Đảm bảo tất cả vector đều là numpy float32
        for k, v in data.items():
            data[k] = np.asarray(v, dtype=np.float32)
        return data

    def save(self) -> None:
        """Lưu database xuống file pickle."""
        with open(self.path, "wb") as f:
            pickle.dump(self.embeddings, f)

    def add(self, label: str, embedding: np.ndarray) -> None:
        """Thêm hoặc cập nhật embedding cho một nhãn."""
        self.embeddings[label] = np.asarray(embedding, dtype=np.float32)

    def get_all(self) -> Dict[str, np.ndarray]:
        """Trả về dict embeddings."""
        return self.embeddings

    def __getitem__(self, label: str) -> np.ndarray:
        return self.embeddings[label]

    def __contains__(self, label: str) -> bool:
        return label in self.embeddings