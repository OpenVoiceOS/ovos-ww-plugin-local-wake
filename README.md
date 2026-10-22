# ovos-ww-plugin-local-wake

An OVOS wake-word plugin for [local-wake](https://github.com/st-matskevich/local-wake). You give it 3 or 4 recordings of a wake word, and it detects that word. It needs no training.

The plugin turns the last second or two of audio into frames of Google's speech-embedding model and compares them with the frames of your recordings by dynamic time warping (DTW). A detection is a DTW distance below 0.16 to any recording.

## Install

```bash
pip install --pre ovos-ww-plugin-local-wake
```

The plugin needs only `onnxruntime` and `numpy`, and carries the 2.4 MB speech-embedding model. It does not install PyTorch, librosa or sounddevice.

## Recordings

Record the wake word 3 or 4 times, with the microphone and in the room that the device uses. Save each recording as a 16-bit WAV file, one wake word per file. Any sample rate works. For example, on Linux:

```bash
arecord -d 2 -r 16000 -c 1 -f S16_LE ~/wakewords/hey_computer/1.wav
```

The plugin cuts the silence before and after the word when it loads the recordings. A recording must hold only the wake word: other sounds in it become part of the match.

## Configuration

```json
"hotwords": {
  "hey_computer": {
    "module": "ovos-ww-plugin-local-wake",
    "samples": "~/wakewords/hey_computer",
    "listen": true
  }
},
"listener": {"wake_word": "hey_computer"}
```

| Key | Default | Meaning |
|---|---|---|
| `samples` | required | Folder or list of WAV recordings of the wake word |
| `threshold` | `0.16` | Largest DTW distance for a detection |
| `buffer_seconds` | longest recording + 0.25 s | Audio compared with the recordings, at least 0.775 s |
| `hop_seconds` | `0.25` | Audio between two comparisons |
| `silence_db` | `20` | A buffer this many dB quieter than the recordings is not compared |

The best threshold depends on the recordings and the microphone. A lower value gives fewer false detections and more misses. The local-wake benchmark found optimal thresholds between 0.14 and 0.21.

## Measured results

Each clip was streamed through the plugin with 1 s of silence before and after it, in chunks of 1024 samples, with the default settings.

With 4 synthetic "alexa" recordings ([synthetic-wakewords](https://huggingface.co/datasets/OpenVoiceOS/synthetic-wakewords)) as references:

| Clips | Detected |
|---|---|
| 116 other synthetic "alexa" | 116 |
| 300 one-second speech clips ([not-wake-words-speech-en](https://huggingface.co/datasets/TigreGotico/not-wake-words-speech-en)) | 1 |
| 60 human "athena" ([community dataset](https://huggingface.co/datasets/OpenVoiceOS/ovos-community-wakewords-dataset)) | 0 |
| 120 human "computer" | 0 |
| 120 synthetic "hey computer" | 0 |

With 2 of the 3 "okay rhasspy" recordings from rhasspy-wake-raven as references, the third gives a distance of 0.15 and speech clips give 0.24 or more.

Noisy recordings do not work. With 4 human recordings from the community dataset as references, the speech clips matched as well as the wake word at every threshold from 0.12 to 0.20. Those recordings are 1.5 to 4 s long, and their background noise is only about 20 dB below the voice, so the noise becomes part of the match. Make your recordings in a quiet room, close to the microphone, with only the wake word in each file. The plugin logs a warning when a recording holds more than 2.5 s of sound.

300 one-second speech clips are not enough to estimate false activations per hour. Measure on your own audio before you use a wake word.

## Limits

- local-wake matches one voice best. The upstream benchmark reports 98.6% detection for the speaker who made the recordings and 81.9% for other speakers, on clean audio.
- The speech-embedding model needs 0.775 s of audio. A shorter wake word is compared together with the sound around it.
- Each comparison runs the model over the whole buffer. Increase `hop_seconds` on slow hardware.

## Credits

local-wake is by Stanislau Matskevich, MIT. The speech-embedding model is by Google, Apache-2.0. See [NOTICE](NOTICE) for the changes.

Report problems at [the issue tracker](https://github.com/OpenVoiceOS/ovos-ww-plugin-local-wake/issues).

## Funding

[![NGI0 Commons Fund](ngi.png)](https://nlnet.nl/project/OpenVoiceOS)

This project was funded through the [NGI0 Commons Fund](https://nlnet.nl/commonsfund), a fund established by [NLnet](https://nlnet.nl) with financial support from the European Commission's [Next Generation Internet](https://ngi.eu) programme, under the aegis of DG Communications Networks, Content and Technology under grant agreement No [101135429](https://cordis.europa.eu/project/id/101135429).
