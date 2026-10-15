# oMLX ASR

Transcribe local audio and video files through oMLX, using Silero VAD to choose chunk boundaries and saving results for resumable runs.

## Requirements

- [uv](https://docs.astral.sh/uv/) and Python 3.12 (`uv sync` can provision Python).
- `ffmpeg` available on `PATH`. On macOS: `brew install ffmpeg`.
- A running oMLX server with a downloaded speech-to-text model, such as `vanch007/SenseVoiceSmall-4bit`.
- Internet access on the first run to download the Silero VAD v6.2.1 ONNX model into `assets/`.

VAD runs locally on CPU using NumPy and ONNX Runtime. Audio chunks are sent to the server configured in `.env` for recognition.

## Setup

From the project directory:

```bash
uv sync --locked
cp .env.example .env
```

Edit `.env` with your server endpoint, API key, and the exact model ID shown by oMLX:

```dotenv
OMLX_ENDPOINT=http://127.0.0.1:8000/v1
OMLX_API_KEY=your-api-key
OMLX_MODEL=SenseVoiceSmall-4bit
```

`OMLX_ENDPOINT` accepts either the server root or a `/v1` base URL. For servers without authentication, set `OMLX_API_KEY=`.
The model must appear in the server's `GET /v1/models` response.

The script reads `.env` beside `transcribe.py`, regardless of the working directory. These settings come from that file, rather than shell environment variables or oMLX's local settings file.

## Usage

Pass the audio or video path as the positional argument. Quote paths containing spaces:

```bash
uv run python transcribe.py "/path/to/video.mp4"
```

Choose an output directory or let the model detect the language:

```bash
uv run python transcribe.py "/path/to/audio.wav" --output-dir output
uv run python transcribe.py "/path/to/audio.wav" --language auto
```

| Argument | Default | Description |
| --- | --- | --- |
| `input` | Required | Path to a media file containing an audio stream |
| `--language` | `zh` | Language hint sent to the ASR model; `auto` requests automatic detection |
| `--chunk-seconds` | `25` | Maximum chunk length in seconds |
| `--output-dir` | `<input_stem>_asr` beside the input | Directory for results and intermediate files; relative paths resolve from the working directory |

Run `uv run python transcribe.py --help` for the command-line options.

## Segmentation and output

The script extracts the first audio stream as 16 kHz mono PCM WAV. With the default 25-second limit, it searches for a non-speech pause between approximately 15 and 25 seconds after each chunk starts. If no suitable pause exists, it cuts near a low speech-probability point and overlaps the next chunk by one second. The final chunk may be shorter.

The full audio timeline is retained, including non-speech sections. At overlapping boundaries, the script removes exact matching suffix/prefix text of up to 50 characters; differently recognized wording may still repeat.

For `/path/to/video.mp4`, results default to `/path/to/video_asr/`:

| File or directory | Contents |
| --- | --- |
| `transcript.txt` | Recognized text, separated by chunk |
| `transcript_timestamped.txt` | Text with chunk start times in `HH:MM:SS` format |
| `results/` | Raw JSON responses for each chunk |
| `segments.json` | Chunk start/end times and overlap flags |
| `manifest.json` | Input path, file metadata, model, language, and segmentation settings |
| `audio.wav`, `chunks/` | Extracted audio and chunk WAV files |
| `vad_probabilities.npy` | Cached VAD speech probabilities |

Timestamps mark chunk starts, not word-level alignment. Intermediate audio files remain on disk for reuse.

## Resume

Repeat the same command with the same input, model, language, chunk length, and output directory. Completed chunk responses are reused, and processing continues from missing results. The oMLX server must still be reachable for the initial model check.

If the input file or recorded settings change, choose a new `--output-dir`.

## Local data

Keep `.env`, input media, model downloads, and transcription output local. The manifest contains the absolute input path; transcripts and chunk JSON contain recognized speech.

The included `.gitignore` excludes environment files (except `.env.example`), common media formats, generated transcription files, `assets/`, `output/`, virtual environments, and editor metadata. Commit `.env.example` with placeholder values and `uv.lock` for reproducible installation.
