# Derived from local-wake (https://github.com/st-matskevich/local-wake),
# Copyright Stanislau Matskevich, MIT. The embedding model is Google's
# speech-embedding (https://www.kaggle.com/models/google/speech-embedding), Apache-2.0.
"""local-wake inference: speech-embedding features compared by DTW with cosine cost."""
from collections.abc import Sequence

import numpy as np
import onnxruntime as ort

SAMPLE_RATE = 16000
# the model emits its first frame after 0.775 s of audio
MIN_SAMPLES = 12400


class SpeechEmbedding:
    """Google's speech-embedding model: 16 kHz float audio to 96-dimensional frames."""

    def __init__(self, model_path: str, providers: Sequence[str] | None = None):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, options, providers=list(providers or ["CPUExecutionProvider"]))

    def features(self, audio: np.ndarray) -> np.ndarray:
        """Frames of shape (96, time) for float32 audio at 16 kHz. Shorter audio is centered in silence."""
        if len(audio) < MIN_SAMPLES:
            padded = np.zeros(MIN_SAMPLES, dtype=np.float32)
            start = (MIN_SAMPLES - len(audio)) // 2
            padded[start:start + len(audio)] = audio
            audio = padded
        emb = self.session.run(None, {"samples:0": audio[None, :].astype(np.float32)})[0]
        return emb[0, :, 0, :].T


def frame_db(audio: np.ndarray, frame: int = 320) -> np.ndarray:
    """Energy in dB of each 20 ms frame."""
    n = max(1, len(audio) // frame)
    frames = np.resize(audio, n * frame).reshape(n, frame) if len(audio) >= frame else audio[None, :]
    return 10 * np.log10(np.mean(np.square(frames, dtype=np.float64), axis=1) + 1e-10)


def trim_silence(audio: np.ndarray, top_db: float = 35.0, margin: int = 1600) -> np.ndarray:
    """Cut the audio to the frames within top_db of the loudest frame, plus a margin.

    When the result is shorter than the model window, it is widened with the
    audio around it, so a short wake word keeps its own background noise
    instead of digital silence.
    """
    frame = 320
    db = frame_db(audio, frame)
    loud = np.flatnonzero(db > db.max() - top_db)
    start = max(0, loud[0] * frame - margin)
    end = min(len(audio), (loud[-1] + 1) * frame + margin)
    if end - start < MIN_SAMPLES:
        grow = MIN_SAMPLES - (end - start)
        start = max(0, start - grow // 2)
        end = min(len(audio), start + MIN_SAMPLES)
        start = max(0, end - MIN_SAMPLES)
    return audio[start:end]


def dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
    """DTW over cosine distances between the columns of x and y, divided by the summed lengths."""
    xn = x / np.maximum(np.linalg.norm(x, axis=0, keepdims=True), 1e-12)
    yn = y / np.maximum(np.linalg.norm(y, axis=0, keepdims=True), 1e-12)
    cost = 1.0 - xn.T @ yn
    n, m = cost.shape
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1])
    return float(acc[n, m] / (n + m))
