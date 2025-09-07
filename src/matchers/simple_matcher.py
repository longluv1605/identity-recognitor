from typing import Dict, Tuple
import numpy as np
from .base import BaseMatcher



def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Tính cosine similarity giữa hai vector."""
    if vec1.size == 0 or vec2.size == 0:
        return 0.0
    dot = float(np.dot(vec1, vec2))
    norm1 = float(np.linalg.norm(vec1))
    norm2 = float(np.linalg.norm(vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

class SimpleMatcher(BaseMatcher):
    """
    Matcher sử dụng phép tính cosine similarity với một dict embeddings
    dạng {label: vector}. Dùng khi số lượng người trong database nhỏ.
    """

    def __init__(self, embeddings: Dict[str, np.ndarray], threshold: float = 0.5):
        super().__init__(threshold=threshold)
        # convert tất cả về mảng float32 để tăng tốc tính toán
        self.embeddings = {k: np.asarray(v, dtype=np.float32) for k, v in embeddings.items()}

    def match(self, query: np.ndarray) -> Tuple[str, float]:
        best_label = "unknown"
        best_score = 0.0
        for label, emb in self.embeddings.items():
            score = cosine_similarity(query, emb)
            if score > best_score:
                best_label = label
                best_score = score
        # So sánh với ngưỡng để quyết định có nhận diện được hay không
        if best_score >= self.threshold:
            return best_label, best_score
        return "unknown", best_score

    def update_database(self, new_embeddings: Dict[str, np.ndarray]) -> None:
        """Bổ sung hoặc ghi đè embeddings trong cơ sở dữ liệu."""
        for label, emb in new_embeddings.items():
            self.embeddings[label] = np.asarray(emb, dtype=np.float32)