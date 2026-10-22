import wave

import numpy as np
import pytest

from ovos_ww_plugin_local_wake import LocalWakeHotwordPlugin, read_wav
from ovos_ww_plugin_local_wake.inference import MIN_SAMPLES, trim_silence


def write_wav(path, audio, rate=16000, channels=1):
    with wave.open(str(path), "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(audio.astype(np.int16).tobytes())


def tone(seconds=0.8, rate=16000, freq=440):
    t = np.arange(int(seconds * rate)) / rate
    return np.sin(2 * np.pi * freq * t) * 8000


def test_samples_are_required():
    with pytest.raises(ValueError, match="needs 'samples'"):
        LocalWakeHotwordPlugin("hey mycroft", {})


def test_missing_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        LocalWakeHotwordPlugin("x", {"samples": str(tmp_path / "nope")})


def test_empty_folder(tmp_path):
    with pytest.raises(ValueError, match="no WAV"):
        LocalWakeHotwordPlugin("x", {"samples": str(tmp_path)})


def test_read_wav_resamples_and_downmixes(tmp_path):
    stereo = np.stack([tone(1.0, 8000), tone(1.0, 8000)], axis=1).reshape(-1)
    write_wav(tmp_path / "a.wav", stereo, rate=8000, channels=2)
    audio = read_wav(str(tmp_path / "a.wav"))
    assert len(audio) == 16000
    assert np.abs(audio).max() == pytest.approx(8000 / 32768, rel=0.01)


def test_folder_loads_every_wav(tmp_path):
    for i in range(3):
        write_wav(tmp_path / f"{i}.wav", tone(freq=300 + 100 * i))
    (tmp_path / "notes.txt").write_text("not audio")
    engine = LocalWakeHotwordPlugin("x", {"samples": str(tmp_path)})
    assert [name for name, _ in engine.references] == ["0.wav", "1.wav", "2.wav"]


def test_detection_clears_buffer_and_latch_is_read_once(tmp_path):
    write_wav(tmp_path / "ref.wav", tone())
    engine = LocalWakeHotwordPlugin("x", {"samples": str(tmp_path), "threshold": 2.0})
    engine.update(tone(0.25).astype(np.int16).tobytes())
    assert engine.found_wake_word() is True
    assert not engine._buffer.any()
    assert engine.found_wake_word() is False


def test_reset(tmp_path):
    write_wav(tmp_path / "ref.wav", tone())
    engine = LocalWakeHotwordPlugin("x", {"samples": str(tmp_path)})
    engine.update(tone(1.0).astype(np.int16).tobytes())
    engine._found = True
    engine.reset()
    assert not engine._buffer.any() and len(engine._pending) == 0
    assert engine.found_wake_word() is False


def test_short_recording_is_padded(tmp_path):
    write_wav(tmp_path / "short.wav", tone(0.3))
    engine = LocalWakeHotwordPlugin("x", {"samples": str(tmp_path)})
    assert engine.references[0][1].shape == (96, 1)


def test_buffer_shorter_than_model_window(tmp_path):
    write_wav(tmp_path / "ref.wav", tone())
    with pytest.raises(ValueError, match="buffer_seconds"):
        LocalWakeHotwordPlugin("x", {"samples": str(tmp_path), "buffer_seconds": 0.5})


def test_buffer_follows_the_longest_recording(tmp_path):
    write_wav(tmp_path / "a.wav", tone(0.6))
    write_wav(tmp_path / "b.wav", tone(1.2))
    assert LocalWakeHotwordPlugin("x", {"samples": str(tmp_path)}).buffer_samples == int(1.2 * 16000) + 4000
    write_wav(tmp_path / "a.wav", tone(0.3))
    write_wav(tmp_path / "b.wav", tone(0.4))
    assert LocalWakeHotwordPlugin("x", {"samples": str(tmp_path)}).buffer_samples == 12400


def test_silence_is_never_compared(tmp_path):
    write_wav(tmp_path / "ref.wav", tone())
    engine = LocalWakeHotwordPlugin("x", {"samples": str(tmp_path), "threshold": 2.0})
    quiet = np.random.default_rng(0).standard_normal(16000 * 2) * 30  # about -60 dBFS
    engine.update(quiet.astype(np.int16).tobytes())
    assert engine.found_wake_word() is False
    assert engine.last_distance == float("inf")


def test_trim_cuts_silence_around_the_word():
    word = (tone(1.0) / 32768).astype(np.float32)
    audio = np.concatenate([np.zeros(32000), word, np.zeros(48000)]).astype(np.float32)
    trimmed = trim_silence(audio)
    assert len(trimmed) == 16000 + 2 * 1600
    assert np.abs(trimmed[1600:1600 + 16000] - word).max() == 0


def test_trim_widens_a_short_word_with_its_own_background():
    rng = np.random.default_rng(1)
    background = (rng.standard_normal(48000) * 0.001).astype(np.float32)
    background[20000:24000] += (tone(0.25) / 32768).astype(np.float32)
    trimmed = trim_silence(background)
    assert len(trimmed) == MIN_SAMPLES
    assert np.count_nonzero(trimmed == 0) == 0
