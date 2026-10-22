"""OVOS wake-word plugin for local-wake, wake words from a few recordings.

The engine turns the last seconds of audio into speech-embedding frames and
compares them with recordings of the wake word by dynamic time warping. A
new wake word needs only 3 or 4 recordings, and no training.

Configuration keys (under the hotword in the ``hotwords`` section of
mycroft.conf):

    samples (str or list):
        Required. A folder of WAV recordings of the wake word, or a list of
        WAV paths. Silence before and after the word is cut when the
        recordings load.

    threshold (float):
        Largest DTW distance that counts as a detection. Default ``0.16``.
        A lower value gives fewer false detections and more misses. The best
        value depends on the recordings and the microphone.

    buffer_seconds (float):
        Audio compared with the recordings. Defaults to the longest
        recording plus 0.25 s, and is never less than 0.775 s.

    hop_seconds (float):
        Audio between two comparisons. Default ``0.25``.

    silence_db (float):
        A buffer whose loudest 20 ms is this many dB below the quietest
        recording's loudest 20 ms is not compared. Default ``20``.
"""
import os
import wave
from os.path import dirname, expanduser, isdir, join
from typing import Any

import numpy as np
from ovos_plugin_manager.templates.hotwords import HotWordEngine
from ovos_utils.log import LOG

from ovos_ww_plugin_local_wake.inference import (
    MIN_SAMPLES,
    SAMPLE_RATE,
    SpeechEmbedding,
    dtw_distance,
    frame_db,
    trim_silence,
)

MODEL_PATH = join(dirname(__file__), "models", "speech-embedding.onnx")


def read_wav(path: str) -> np.ndarray:
    """Read a 16-bit PCM WAV file as mono float32 at 16 kHz."""
    with wave.open(path, "rb") as f:
        if f.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit PCM WAV is supported")
        rate, channels = f.getframerate(), f.getnchannels()
        audio = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE:
        n_out = int(len(audio) * SAMPLE_RATE / rate)
        audio = np.interp(np.arange(n_out) / SAMPLE_RATE, np.arange(len(audio)) / rate, audio).astype(np.float32)
    return audio


class LocalWakeHotwordPlugin(HotWordEngine):
    """Wake-word detection by comparison with recordings of the wake word."""

    def __init__(self, key_phrase: str = "hey mycroft", config: dict[str, Any] | None = None):
        super().__init__(key_phrase, config)
        self.threshold = float(self.config.get("threshold", 0.16))
        self.hop_samples = max(1, int(float(self.config.get("hop_seconds", 0.25)) * SAMPLE_RATE))
        self.model = SpeechEmbedding(MODEL_PATH)
        self.references, longest, quietest_peak = self._load_references()
        self.min_peak_db = quietest_peak - float(self.config.get("silence_db", 20.0))
        if "buffer_seconds" in self.config:
            self.buffer_samples = int(float(self.config["buffer_seconds"]) * SAMPLE_RATE)
            if self.buffer_samples < MIN_SAMPLES:
                raise ValueError(f"buffer_seconds must be at least {MIN_SAMPLES / SAMPLE_RATE}")
        else:
            self.buffer_samples = max(MIN_SAMPLES, longest + SAMPLE_RATE // 4)

        self._buffer = np.zeros(self.buffer_samples, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)
        self._found = False
        self.last_distance = float("inf")

    def _load_references(self) -> tuple[list[tuple[str, np.ndarray]], int, float]:
        """Features of each recording, the longest recording in samples, and the quietest recording's peak in dB."""
        samples = self.config.get("samples")
        if not samples:
            raise ValueError(f"local-wake needs 'samples' for '{self.key_phrase}': "
                             "a folder or a list of WAV recordings of the wake word")
        if isinstance(samples, str):
            folder = expanduser(samples)
            if not isdir(folder):
                raise FileNotFoundError(f"samples folder not found: {folder}")
            samples = sorted(join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".wav"))
        if not samples:
            raise ValueError(f"no WAV recordings found for '{self.key_phrase}'")
        audio = [(os.path.basename(p), trim_silence(read_wav(expanduser(p)))) for p in samples]
        for name, a in audio:
            if len(a) > 2.5 * SAMPLE_RATE:
                LOG.warning(f"local-wake: {name} holds {len(a) / SAMPLE_RATE:.1f} s of sound after trimming; "
                            "a recording should hold only the wake word")
        references = [(name, self.model.features(a)) for name, a in audio]
        LOG.info(f"local-wake loaded {len(references)} recordings for '{self.key_phrase}'")
        return references, max(len(a) for _, a in audio), min(frame_db(a).max() for _, a in audio)

    def update(self, chunk: bytes):
        audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768
        self._pending = np.concatenate([self._pending, audio])
        while len(self._pending) >= self.hop_samples:
            hop, self._pending = self._pending[:self.hop_samples], self._pending[self.hop_samples:]
            self._buffer = np.concatenate([self._buffer[len(hop):], hop])
            if frame_db(self._buffer).max() < self.min_peak_db:
                self.last_distance = float("inf")
                continue
            features = self.model.features(self._buffer)
            self.last_distance = min(dtw_distance(features, ref) for _, ref in self.references)
            if self.last_distance < self.threshold:
                LOG.debug(f"local-wake detected '{self.key_phrase}' at distance {self.last_distance:.4f}")
                self._found = True
                # the buffer still holds the wake word: clear it so the next hops do not match it again
                self._buffer[:] = 0

    def found_wake_word(self) -> bool:
        found, self._found = self._found, False
        return found

    def reset(self):
        self._buffer[:] = 0
        self._pending = np.zeros(0, dtype=np.float32)
        self._found = False
