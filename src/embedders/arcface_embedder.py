import os
import cv2
import numpy as np
import onnxruntime as ort
from typing import Optional, Tuple, List

from .base import BaseEmbedder

class ArcFaceEmbedder(BaseEmbedder):
    """
    ArcFace Model for Face Recognition

    This class implements a face encoder using the ArcFace architecture,
    loading a pre-trained model from an ONNX file.
    """

    def __init__(self, model_path: str, input_size: Tuple[int, int] = (112, 112), normalized: bool = True) -> None:
        """
        Initializes the ArcFace face encoder model.

        Args:
            model_path (str): Path to ONNX model file.
            input_size (tuple(int)): 

        Raises:
            RuntimeError: If model initialization fails.
        """
        self.model_path = model_path
        self.input_size = input_size
        self.normalization_mean = 127.5
        self.normalization_scale = 127.5
        self.normalized = normalized

        try:
            self.session = ort.InferenceSession(
                self.model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )

            input_config = self.session.get_inputs()[0]
            self.input_name = input_config.name

            input_shape = input_config.shape
            model_input_size = tuple(input_shape[2:4][::-1])
            if model_input_size != self.input_size:
                print(
                    f"Model input size {model_input_size} differs from configured size {self.input_size}"
                )

            self.output_names = [o.name for o in self.session.get_outputs()]
            self.output_shape = self.session.get_outputs()[0].shape
            self.embedding_size = self.output_shape[1]

            assert len(self.output_names) == 1, "Expected only one output node."

        except Exception as e:
            raise RuntimeError(f"Failed to initialize model session for '{self.model_path}'") from e

    def _preprocess(self, face: np.ndarray) -> np.ndarray:
        """
        Preprocess the face image: resize, normalize, and convert to the required format.

        Args:
            face_image (np.ndarray): Input face image in BGR format.

        Returns:
            np.ndarray: Preprocessed image blob ready for inference.
        """
        resized_face = cv2.resize(face, self.input_size)

        if isinstance(self.normalization_scale, (list, tuple)):
            mean_array = np.array(self.normalization_mean, dtype=np.float32)
            scale_array = np.array(self.normalization_scale, dtype=np.float32)
            normalized_face = (face - mean_array) / scale_array

            # Change to NCHW format (batch, channels, height, width)
            transposed_face = np.transpose(normalized_face, (2, 0, 1))
            face_blob = np.expand_dims(transposed_face, axis=0)
        else:
            # Single-value normalization using cv2.dnn
            face_blob = cv2.dnn.blobFromImage(
                resized_face,
                scalefactor=1.0 / self.normalization_scale,
                size=self.input_size,
                mean=(self.normalization_mean,)*3,
                swapRB=True
            )
        return face_blob

    def embed(
        self,
        face: np.ndarray,
    ) -> np.ndarray:

        try:
            face_blob = self._preprocess(face)
            embedding = self.session.run(self.output_names, {self.input_name: face_blob})[0]

            if self.normalized:
                # L2 normalization of embedding
                norm = np.linalg.norm(embedding, axis=1, keepdims=True)
                normalized_embedding = embedding / norm
                return normalized_embedding.flatten()

            return embedding.flatten()

        except Exception as e:
            print(f"Error extracting face embedding: {e}")
            raise