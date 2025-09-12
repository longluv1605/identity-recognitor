"""Evaluate embeddings database to derive similarity/distance thresholds.

Usage (example):
	python scripts/evaluate.py --config config/config.yaml --mode cosine --target_far 0.01

Outputs:
	Prints stats + saves JSON (optional) with suggested thresholds.
"""

import os
import argparse
import json
import math
from itertools import combinations
import numpy as np
import yaml
import sys

##############
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
##############

from src.database import EmbeddingDatabase
from src.matchers.simple_matcher import cosine_similarity, l2_distance


def load_db(path: str) -> dict:
	if not os.path.exists(path):
		raise FileNotFoundError(f"Database not found: {path}")
	db = EmbeddingDatabase(path)
	return db.get_all()


def pairwise_scores(db: dict, mode: str = "cosine"):
	pos = []  # same identity
	neg = []  # different identity
	labels = list(db.keys())
	for label in labels:
		emb_array = np.asarray(db[label], dtype=np.float32)
		# ensure 2D
		if emb_array.ndim == 1:
			emb_array = emb_array.reshape(1, -1)
		# normalise for cosine
		if mode == "cosine":
			norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
			emb_array = emb_array / np.clip(norms, 1e-12, None)
		# positives
		for (i, j) in combinations(range(emb_array.shape[0]), 2):
			if mode == "cosine":
				pos.append(cosine_similarity(emb_array[i], emb_array[j]))
			else:
				pos.append(l2_distance(emb_array[i], emb_array[j]))
		# store back
		db[label] = emb_array

	# negatives
	for (la, lb) in combinations(labels, 2):
		a = db[la]
		b = db[lb]
		for va in a:
			for vb in b:
				if mode == "cosine":
					neg.append(cosine_similarity(va, vb))
				else:
					neg.append(l2_distance(va, vb))
	return np.array(pos, dtype=np.float32), np.array(neg, dtype=np.float32)


def eer_threshold(pos: np.ndarray, neg: np.ndarray, mode: str):
	"""Compute approximate EER threshold.

	For cosine (higher=match): vary t; FAR = fraction(neg >= t), FRR = fraction(pos < t)
	For distance (lower=match): FAR = fraction(neg <= t), FRR = fraction(pos > t)
	"""
	if pos.size == 0 or neg.size == 0:
		return math.nan
	if mode == "cosine":
		ts = np.linspace(-1, 1, 200)
		best_t = 0
		best_diff = 1e9
		for t in ts:
			far = (neg >= t).mean()
			frr = (pos < t).mean()
			d = abs(far - frr)
			if d < best_diff:
				best_diff = d
				best_t = t
		return best_t
	else:
		ts = np.linspace(0, float(max(pos.max(), neg.max(), 1.0)), 200)
		best_t = 0
		best_diff = 1e9
		for t in ts:
			far = (neg <= t).mean()
			frr = (pos > t).mean()
			d = abs(far - frr)
			if d < best_diff:
				best_diff = d
				best_t = t
		return best_t


def threshold_for_far(pos: np.ndarray, neg: np.ndarray, mode: str, target_far: float):
	if pos.size == 0 or neg.size == 0:
		return math.nan
	if mode == "cosine":
		# Want FAR = fraction(neg >= t) <= target_far => choose t = quantile(1 - target_far)
		q = np.quantile(neg, 1 - target_far)
		return float(q)
	else:
		# FAR = fraction(neg <= t) <= target_far => choose t = quantile(target_far)
		q = np.quantile(neg, target_far)
		return float(q)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--config", default="config/config.yaml")
	ap.add_argument("--mode", choices=["cosine", "l2"], default="cosine")
	ap.add_argument("--target_far", type=float, default=0.01)
	ap.add_argument("--save_json", default=None, help="Path to save stats JSON")
	args = ap.parse_args()

	with open(args.config, 'r', encoding='utf-8') as f:
		cfg = yaml.safe_load(f)
	db_path = cfg['database']['output']
	db = load_db(db_path)

	pos, neg = pairwise_scores(db, mode=args.mode)
	print(f"Positive pairs: {pos.size} | Negative pairs: {neg.size}")
	if pos.size == 0 or neg.size == 0:
		print("Not enough pairs to evaluate.")
		return

	stats = {
		'mode': args.mode,
		'pos_mean': float(pos.mean()),
		'pos_std': float(pos.std()),
		'neg_mean': float(neg.mean()),
		'neg_std': float(neg.std()),
		'pos_p05': float(np.quantile(pos, 0.05)),
		'pos_p50': float(np.quantile(pos, 0.50)),
		'pos_p95': float(np.quantile(pos, 0.95)),
		'neg_p05': float(np.quantile(neg, 0.05)),
		'neg_p50': float(np.quantile(neg, 0.50)),
		'neg_p95': float(np.quantile(neg, 0.95)),
	}

	t_eer = eer_threshold(pos, neg, mode=args.mode)
	t_far = threshold_for_far(pos, neg, mode=args.mode, target_far=args.target_far)
	stats['threshold_eer'] = t_eer
	stats['threshold_target_far'] = t_far
	stats['target_far'] = args.target_far

	print("=== Statistics ===")
	for k, v in stats.items():
		print(f"{k}: {v}")

	if args.save_json:
		with open(args.save_json, 'w', encoding='utf-8') as f:
			json.dump(stats, f, indent=2)
		print(f"Saved stats to {args.save_json}")


if __name__ == "__main__":
	main()

