import os
from typing import List, Tuple, Optional

import cv2
import numpy as np
import onnxruntime as ort

from .base import BaseDetector

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]  # current shape [h, w]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

def xywh2xyxy(x):
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
    return y

def nms(boxes, scores, iou_thres=0.45):
    if len(boxes) == 0:
        return []
    boxes = boxes.astype(np.float32)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return keep

class YoloOnnxDetector(BaseDetector):
    """
    YOLOv8-face ONNX detector dùng onnxruntime.
    Trả về: List[(x, y, w, h, conf, cls)]
    """

    def __init__(self,
                 model_path: str,
                 device: str = "cpu",
                 input_size: int = 640,
                 conf_thres: float = 0.25,
                 iou_thres: float = 0.45,
                 providers: Optional[List[str]] = None):
        super().__init__()
        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        if providers is None:
            if device == "cuda":
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            elif device.lower() in ("dml", "directml"):
                providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(model_path, sess_options=so, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = int(input_size)
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)

        outs = self.session.get_outputs()
        self.output_name = outs[0].name  # thường chỉ 1 output

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float, int]]:
        img0 = frame  # BGR
        img, r, (dw, dh) = letterbox(img0, new_shape=(self.input_size, self.input_size), stride=32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # 3xHxW
        img = np.expand_dims(img, 0).copy()  # 1x3xHxW

        preds = self.session.run([self.output_name], {self.input_name: img})[0]
        # Chuẩn hóa shape: (Npred, no)
        if preds.ndim == 3:
            if preds.shape[1] < preds.shape[2]:
                preds = np.squeeze(preds, 0).T  # (no, N) -> (N, no)
            else:
                preds = np.squeeze(preds, 0)    # (N, no)
        # giờ preds: (N, no)
        if preds.ndim != 2 or preds.shape[1] < 5:
            return []

        boxes = preds[:, :4]
        obj = preds[:, 4]
        if preds.shape[1] > 5:
            cls_scores = preds[:, 5:]
            cls_ids = cls_scores.argmax(axis=1)
            cls_conf = cls_scores.max(axis=1)
            scores = obj * cls_conf
        else:
            cls_ids = np.zeros((preds.shape[0],), dtype=np.int32)
            scores = obj

        # lọc theo conf
        mask = scores >= self.conf_thres
        if not np.any(mask):
            return []
        boxes = boxes[mask]
        scores = scores[mask]
        cls_ids = cls_ids[mask]

        # xywh -> xyxy (trên ảnh sau letterbox)
        boxes_xyxy = xywh2xyxy(boxes)

        # bỏ padding và scale về ảnh gốc
        boxes_xyxy[:, [0, 2]] -= dw
        boxes_xyxy[:, [1, 3]] -= dh
        boxes_xyxy /= r

        # clamp
        h0, w0 = img0.shape[:2]
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, w0 - 1)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, h0 - 1)

        # NMS
        keep = nms(boxes_xyxy, scores, self.iou_thres)
        boxes_xyxy = boxes_xyxy[keep]
        scores = scores[keep]
        cls_ids = cls_ids[keep]

        out = []
        for (x1, y1, x2, y2), s, c in zip(boxes_xyxy, scores, cls_ids):
            x, y = int(round(x1)), int(round(y1))
            w, h = int(round(x2 - x1)), int(round(y2 - y1))
            out.append((x, y, w, h, float(s), int(c)))
        return out