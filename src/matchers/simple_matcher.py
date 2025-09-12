from typing import Dict, Tuple, List
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
    """Matcher với chiến lược aggregation & margin open-set rejection."""

    def __init__(self,
                 embeddings: Dict[str, np.ndarray],
                 threshold: float = 0.6,
                 use_cosine: bool = True,
                 aggregation: str = "topk-mean",
                 topk: int = 3,
                 second_best_margin: float = 0.05,
                 normalize_query: bool = True):
        super().__init__(threshold=threshold)
        self.embeddings: Dict[str, np.ndarray] = {}
        for k, v in embeddings.items():
            v = np.asarray(v, dtype=np.float32)
            if v.ndim == 1:
                v = v.reshape(1, -1)
            elif v.ndim != 2:
                raise ValueError(f"Invalid embedding shape for {k}: {v.shape}")
            # assume already normalised but ensure
            norms = np.linalg.norm(v, axis=1, keepdims=True)
            v = v / np.clip(norms, 1e-12, None)
            self.embeddings[k] = v
        self.use_cosine = use_cosine
        self.aggregation = aggregation
        self.topk = max(1, topk)
        self.second_best_margin = second_best_margin
        self.normalize_query = normalize_query

    # ---------------- Aggregation helpers -----------------
    def _aggregate(self, scores: List[float]) -> float:
        if not scores:
            return 0.0 if self.use_cosine else float('inf')
        arr = np.asarray(scores, dtype=np.float32)
        if self.use_cosine:
            if self.aggregation == "mean":
                return float(arr.mean())
            if self.aggregation == "median":
                return float(np.median(arr))
            if self.aggregation.startswith("topk"):
                k = min(self.topk, arr.size)
                return float(np.sort(arr)[-k:].mean())
            return float(arr.max())  # default max
        # distance mode
        if self.aggregation == "mean":
            return float(arr.mean())
        if self.aggregation == "median":
            return float(np.median(arr))
        if self.aggregation.startswith("topk"):
            k = min(self.topk, arr.size)
            return float(np.sort(arr)[:k].mean())
        return float(arr.min())

    def match(self, query: np.ndarray) -> Tuple[str, float]:
        if query.size == 0:
            return "unknown", 0.0 if self.use_cosine else float('inf')
        q = query.astype(np.float32)
        if self.use_cosine and self.normalize_query:
            n = np.linalg.norm(q)
            if n > 0:
                q = q / n

        scores_per_identity = []  # (label, agg_score)
        for label, emb_array in self.embeddings.items():
            scores = []
            for emb in emb_array:
                if self.use_cosine:
                    s = cosine_similarity(q, emb)
                else:
                    s = l2_distance(q, emb)
                scores.append(s)
            agg = self._aggregate(scores)
            scores_per_identity.append((label, agg))

        if not scores_per_identity:
            return "unknown", 0.0 if self.use_cosine else float('inf')

        # sort: cosine desc, distance asc
        reverse = self.use_cosine
        scores_per_identity.sort(key=lambda x: x[1], reverse=reverse)
        best_label, best_score = scores_per_identity[0]
        second_score = scores_per_identity[1][1] if len(scores_per_identity) > 1 else (0.0 if self.use_cosine else float('inf'))

        if self.use_cosine:
            if best_score < self.threshold:
                return "unknown", best_score
            if (best_score - second_score) < self.second_best_margin:
                return "unknown", best_score
            return best_label, best_score
        else:
            if best_score > self.threshold:
                return "unknown", best_score
            if (second_score - best_score) < self.second_best_margin:
                return "unknown", best_score
            return best_label, best_score

    def update_database(self, new_embeddings: Dict[str, np.ndarray]) -> None:
        for label, emb in new_embeddings.items():
            emb = np.asarray(emb, dtype=np.float32)
            if emb.ndim == 1:
                emb = emb.reshape(1, -1)
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / np.clip(norms, 1e-12, None)
            self.embeddings[label] = emb