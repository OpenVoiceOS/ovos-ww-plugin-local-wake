"""Wake-word detection through the OPM factory and ovoscope's MiniListener, on real audio.

Clips (see test/clips/README.md): okay_rhasspy/ are human recordings from
local-wake; alexa_synthetic/ comes from OpenVoiceOS/synthetic-wakewords and
negative/ from TigreGotico/not-wake-words-speech-en (both CC-BY-4.0).
"""
from pathlib import Path

import numpy as np
from ovos_plugin_manager.wakewords import OVOSWakeWordFactory, find_wake_word_plugins
from ovoscope.listener import get_mini_listener

from ovos_ww_plugin_local_wake import read_wav

CLIPS = Path(__file__).parent / "clips"
PLUGIN = "ovos-ww-plugin-local-wake"
RHASSPY = sorted((CLIPS / "okay_rhasspy").glob("*.wav"))
SYNTHETIC = sorted((CLIPS / "alexa_synthetic").glob("*.wav"))
NEGATIVES = sorted((CLIPS / "negative").glob("*.wav"))


def pcm(*paths, gap=0.5):
    silence = np.zeros(int(gap * 16000), dtype=np.float32)
    parts = [silence]
    for p in paths:
        parts += [read_wav(str(p)), silence]
    return (np.concatenate(parts) * 32767).astype(np.int16).tobytes()


def scan(samples, audio):
    config = {"hotwords": {"wake": {"module": PLUGIN, "samples": [str(p) for p in samples]}}}
    engine = OVOSWakeWordFactory.create_hotword("wake", config)
    assert type(engine).__module__.startswith("ovos_ww_plugin_local_wake")
    listener = get_mini_listener(ww_instances={"wake": engine})
    try:
        return listener.scan_for_wakeword(audio, frame_size=2048)
    finally:
        listener.shutdown()


def test_entry_point_is_discovered():
    assert PLUGIN in find_wake_word_plugins()


def test_human_recordings_detect_the_same_speaker():
    detected, frame = scan(RHASSPY[1:], pcm(RHASSPY[0]))
    assert detected
    assert frame * 1024 / 16000 > 0.5


def test_human_recordings_ignore_other_speech():
    assert scan(RHASSPY, pcm(*NEGATIVES)) == (False, None)


def test_synthetic_recordings_detect_a_new_voice():
    detected, _ = scan(SYNTHETIC[:4], pcm(SYNTHETIC[4]))
    assert detected


def test_synthetic_recordings_ignore_other_speech():
    assert scan(SYNTHETIC[:4], pcm(*NEGATIVES)) == (False, None)
