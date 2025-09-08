from typing import Dict, Tuple
import numpy as np
from .base import BaseMatcher


def l2_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Tính L2 distance giữa hai vector. Trả về distance (càng nhỏ càng giống)."""
    if vec1.size == 0 or vec2.size == 0:
        return float('inf')
    return float(np.linalg.norm(vec1 - vec2))

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
    Matcher hỗ trợ nhiều embeddings cho mỗi người.
    embeddings có thể là:
    - Dict[str, np.ndarray]: 1 embedding per person (legacy)
    - Dict[str, np.ndarray]: nhiều embeddings per person (shape: [n_embeddings, embedding_dim])
    """

    def __init__(self, embeddings: Dict[str, np.ndarray], threshold: float = 1.2, use_cosine: bool = False):
        super().__init__(threshold=threshold)
        # Xử lý cả single embedding và multiple embeddings
        self.embeddings = {}
        for k, v in embeddings.items():
            v = np.asarray(v, dtype=np.float32)
            if v.ndim == 1:
                # Single embedding - giữ nguyên
                self.embeddings[k] = v.reshape(1, -1)  # [1, embedding_dim]
            elif v.ndim == 2:
                # Multiple embeddings - đã đúng format
                self.embeddings[k] = v  # [n_embeddings, embedding_dim]
            else:
                raise ValueError(f"Invalid embedding shape for {k}: {v.shape}")
        self.use_cosine = use_cosine

    def match(self, query: np.ndarray) -> Tuple[str, float]:
        best_label = "unknown"
        
        if self.use_cosine:
            # Sử dụng cosine similarity (score cao = giống)
            best_score = 0.0
            for label, emb_array in self.embeddings.items():
                # Tính similarity với tất cả embeddings của người này
                scores = []
                for emb in emb_array:
                    score = cosine_similarity(query, emb)
                    scores.append(score)
                # Lấy score cao nhất (best match)
                max_score = max(scores)
                if max_score > best_score:
                    best_label = label
                    best_score = max_score
            # So sánh với ngưỡng
            if best_score >= self.threshold:
                return best_label, best_score
            return "unknown", best_score
        else:
            # Sử dụng L2 distance (score nhỏ = giống)
            best_distance = float('inf')
            for label, emb_array in self.embeddings.items():
                # Tính distance với tất cả embeddings của người này
                distances = []
                for emb in emb_array:
                    distance = l2_distance(query, emb)
                    distances.append(distance)
                # Lấy distance nhỏ nhất (best match)
                min_distance = min(distances)
                if min_distance < best_distance:
                    best_label = label
                    best_distance = min_distance
            # So sánh với ngưỡng (distance nhỏ hơn threshold = trùng khớp)
            if best_distance <= self.threshold:
                return best_label, best_distance
            return "unknown", best_distance

    def update_database(self, new_embeddings: Dict[str, np.ndarray]) -> None:
        """Bổ sung hoặc ghi đè embeddings trong cơ sở dữ liệu."""
        for label, emb in new_embeddings.items():
            emb = np.asarray(emb, dtype=np.float32)
            if emb.ndim == 1:
                self.embeddings[label] = emb.reshape(1, -1)
            else:
                self.embeddings[label] = emb