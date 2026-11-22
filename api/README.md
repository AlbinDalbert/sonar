# sonar api

FastAPI wrapper around **Qwen3-TTS** and **Qwen3-ASR**: speech synthesis, streaming, voice cloning, voice design, and speech-to-text transcription.

- Default port: `8888`
- Default response format: `wav`

## Endpoints

### `POST /api/tts`
Synthesize speech.

Request JSON:
```json
{
  "text": "Hello from Qwen3-TTS",
  "language": "English",
  "audio_format": "wav",
  "mode": "design",
  "model_size": "quality",
  "cpu_mode": false,
  "voice_preset": "narrator",
  "voice_description": "Optional explicit design description",
  "reference_audio": "myvoice.wav"
}
```

### `POST /api/tts/audio-stream`
Stream audio **while it is being generated**.

Codec frames are captured from the model as they are sampled (12.5 per second of audio) and decoded in small spans by a stateful decoder (conv/overlap state + KV cache carried between spans, so each frame is decoded once and the audio is identical to a full decode), so the first audio arrives after about 4 frames (~320 ms of audio) plus prompt prefill, instead of after a whole sentence. Long text is still split into chunks and generated one after another on the same stream, so memory stays flat.

Same request JSON as `/api/tts`, with `audio_format`:
- `pcm`: raw signed 16-bit little-endian mono samples. Easiest to play progressively with Web Audio.
- `wav`: the same samples behind a streaming WAV header (sizes `0xffffffff`). Plays in `<audio>`/players and with curl.

Response headers: `X-Sample-Rate` (24000 for the 12Hz models), `X-Channels: 1`, `X-Sample-Format: s16le`. These are CORS-exposed, so `fetch()` clients can read them.

Errors before the first audio (bad voice, GPU busy, ...) return a normal JSON error status. Closing the connection stops generation at the next frame.

Example:
```bash
curl -N -X POST http://localhost:8888/api/tts/audio-stream \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello from streamed Qwen3-TTS","mode":"design","voice_description":"Calm, warm, mid-pitch narrator","audio_format":"wav"}' \
  --output stream.wav
```

Live playback test: `... -d '{..., "audio_format":"pcm"}' | ffplay -f s16le -ar 24000 -ac 1 -nodisp -`

This relies on internals of the pinned `qwen-tts==0.1.1` (the talker forward patch and the 12Hz decoder). Re-check it if that pin ever changes.

### `POST /api/tts/stream`
Server-sent events with progress and a final base64 WAV payload. Useful for UI progress, but not memory-efficient for long text; prefer `/api/tts/audio-stream` for long text.

### `POST /v1/audio/speech`
OpenAI-compatible streaming text-to-speech endpoint backed by Qwen3-TTS. Streams audio incrementally as codec frames are generated (same frame-streaming pipeline as `/api/tts/audio-stream`).

Request JSON:
```json
{
  "model": "tts-1",
  "input": "Hello from Qwen3-TTS",
  "voice": "alloy",
  "response_format": "mp3",
  "speed": 1.0,
  "stream_format": "audio"
}
```

Fields:
- `input` (required): text to synthesize
- `model` (optional, default `tts-1`): `tts-1-hd`, `quality`, or anything containing `1.7b` selects the `1.7B` clone model; otherwise `0.6B` (`design` mode always uses `1.7B-VoiceDesign`)
- `voice` (optional, default `alloy`): resolved in this order:
  1. Explicit prefixes: `preset:<name>`, `clone:<filename>`, or `design:<description>`
  2. Matching voice preset name from `/voice-presets` (`design` mode)
  3. Matching reference audio filename or stem from `/reference-audio` (`clone` mode; e.g. `myvoice.wav` or `myvoice`)
  4. Standard OpenAI voice name (`alloy`, `ash`, `ballad`, `coral`, `echo`, `fable`, `onyx`, `nova`, `sage`, `shimmer`, `verse`, `marin`, `cedar`) or `default`. Uses `QWEN_OPENAI_DEFAULT_VOICE` if set, otherwise a built-in `design` description.
  5. Multi-word plain text treated as a `design` voice description
- `instructions` (optional): appended to the voice description in `design` mode (ignored in `clone` mode)
- `response_format` (optional, default `mp3`): `mp3`, `opus`, `aac`, `flac`, `wav`, or `pcm` (24 kHz mono signed 16-bit little-endian)
- `speed` (optional, default `1.0`): `0.25` to `4.0`
- `stream_format` (optional, default `audio`):
  - `audio`: streams raw audio bytes in the requested `response_format`
  - `sse`: streams Server-Sent Events (`speech.audio.delta` with base64 audio chunks, then `speech.audio.done` with usage)
- Optional Sonar extensions: `language` (default `Auto`), `user` (default `default`), `cpu_mode` (default `false`)

Examples:
```bash
# Stream MP3 directly
curl -N -X POST http://localhost:8888/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"tts-1","voice":"alloy","input":"Hello from Sonar","response_format":"mp3"}' \
  --output speech.mp3

# Stream PCM via SSE (OpenAI gpt-4o-mini-tts style)
curl -N -X POST http://localhost:8888/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-4o-mini-tts","voice":"coral","instructions":"Speak calmly","input":"Hello via SSE","response_format":"pcm","stream_format":"sse"}'
```

Fields for `/api/tts` and `/api/tts/stream`:
- `text` (required)
- `language` (default `Auto`)
- `audio_format` (`wav` or `ogg`)
- `mode` (`clone` or `design`)
- `model_size` (`fast` or `quality`, default `fast`; `design` always uses `quality`)
- `reference_audio` (required in `clone` mode)
- `voice_description` (required in `design` mode unless `voice_preset` is provided)
- `voice_preset` (optional preset name from `/voice-presets`; overrides `voice_description` and forces `design` mode)
- `cpu_mode` (default `false`; when `true`, runs the complete TTS pipeline on CPU with no GPU fallback)

### `POST /api/asr`
Transcribe an uploaded audio file with **Qwen3-ASR** (`Qwen/Qwen3-ASR-0.6B` or `Qwen/Qwen3-ASR-1.7B`).

Multipart form fields:
- `file` (required; WAV, MP3, M4A, OGG/Opus, WebM, FLAC, or any format `ffmpeg` can decode)
- `model_size` (`fast` or `quality`, default `fast` = `0.6B`; `quality` = `1.7B`)
- `language` (optional; omit or pass `Auto` for automatic detection; accepts full names like `English` or ISO codes like `en`, `sv`)
- `context` (optional domain vocabulary / prompt hint)
- `cpu_mode` (default `false`; explicit CPU-only execution, no automatic fallback)

Example:
```bash
curl -X POST http://localhost:8888/api/asr \
  -F "file=@recording.m4a" \
  -F "model_size=fast"
```

Response JSON:
```json
{
  "text": "Hello from Qwen3-ASR.",
  "language": "English",
  "model_id": "Qwen/Qwen3-ASR-0.6B",
  "model_size": "fast",
  "duration_s": 2.41,
  "t_total_ms": 412,
  "t_transcribe_ms": 380
}
```

### `POST /v1/audio/transcriptions`
OpenAI Whisper-compatible transcription endpoint backed by Qwen3-ASR for drop-in use with external apps.

Multipart form fields:
- `file` (required)
- `model` (optional; `quality` or anything containing `1.7b` selects `Qwen/Qwen3-ASR-1.7B`, otherwise `Qwen/Qwen3-ASR-0.6B`, including `whisper-1`)
- `language` (optional; ISO-639 codes like `en`/`sv` or full language names)
- `prompt` (optional; forwarded as Qwen3-ASR `context`)
- `response_format` (`json`, `text`, or `verbose_json`; default `json`)
- `cpu_mode` (optional boolean, default `false`)

Example:
```bash
curl -X POST http://localhost:8888/v1/audio/transcriptions \
  -F "file=@recording.wav" \
  -F "model=whisper-1"
```

### Errors

If CUDA runs out of VRAM while loading or running a requested model, the API returns HTTP `503` with a retry hint (or an equivalent error event if streaming headers were already sent):

```json
{"detail":"GPU busy"}
```

CPU mode is explicit only: the API never falls back from GPU to CPU automatically.

### `GET /healthz`
Process health.

### `GET /readyz`
Model readiness.

### `GET /info`
Returns the model IDs, modes, sizes, loaded model, supported languages and generation settings.

### Reference-audio management
- `GET /reference-audio` → list uploaded files
- `POST /reference-audio/upload` → multipart upload (`file` form field)
- `DELETE /reference-audio/{filename}` → remove file

### Voice design presets
- `GET /voice-presets` → list presets (`presets[]`)
- `POST /voice-presets` → create/update preset
- `DELETE /voice-presets/{name}` → delete preset

Presets are stored per user (`?user=name`, default `default`) in `/reference-audio/<user>/voice_presets.json`. None are built in.

## Voice cloning behavior

When `reference_audio` is set in `/api/tts`, the API validates `/reference-audio/<filename>` and calls:
- `model.generate_voice_clone(..., ref_audio=<path>, x_vector_only_mode=True)`

This follows the Qwen3-TTS voice clone API (`ref_audio` parameter), allowing single-shot clone usage without `ref_text`.

## Local run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install --no-deps qwen-tts==0.1.1

python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8888
```

## Environment variables

- `QWEN_DEVICE`
- `QWEN_DTYPE`
- `QWEN_ATTN_IMPL` (default `sdpa`; use `eager` if SDPA causes trouble)
- `QWEN_CPU_DTYPE` (default `float32`; used when request `cpu_mode=true`)
- `QWEN_CPU_ATTN_IMPL` (default `sdpa`; used when request `cpu_mode=true`)
- `QWEN_IDLE_UNLOAD_SECONDS` (seconds to keep a loaded model warm after a request finishes before unloading it, default `5`; set `0` to unload immediately)
- `QWEN_SPEAKER_CACHE_MAX_ITEMS` (in-memory LRU cache size for extracted clone speaker embeddings, default `64`; set `0` to disable)
- `QWEN_ASR_MODEL_ID_06` (default `Qwen/Qwen3-ASR-0.6B`)
- `QWEN_ASR_MODEL_ID_17` (default `Qwen/Qwen3-ASR-1.7B`)
- `QWEN_ASR_MAX_NEW_TOKENS` (max output tokens per ASR chunk, default `2048`)
- `QWEN_ASR_MAX_INFERENCE_BATCH_SIZE` (internal ASR batch cap for >20 min split audio, default `1`)
- `QWEN_ASR_MAX_UPLOAD_MB` (maximum ASR audio upload size in MB, default `100`)
- `QWEN_MAX_CHUNK_CHARS` (text chunk size for long text/streaming, default `160`)
- `QWEN_STREAM_FIRST_FRAMES` (audio-stream: frames to collect before the first decode, default `4` = 320 ms)
- `QWEN_STREAM_STEP_FRAMES` (audio-stream: minimum frames per later decode, default `4`; a slow decoder automatically batches more)
- `QWEN_OPENAI_DEFAULT_VOICE` (optional fallback voice for `/v1/audio/speech` when a standard OpenAI voice name or `default` has no local preset or reference audio match; accepts a preset name, reference audio filename, prefixed voice, or description)
- `QWEN_CORS_ORIGINS` (comma-separated, default `*`)
- `QWEN_TRIM_CHUNK_SILENCE` (default `1`; trims generated leading/trailing silence at chunk boundaries)
- `QWEN_TRIM_SILENCE_KEEP_MS` (default `60`)
- `QWEN_CHUNK_FADE_MS` (default `8`)
- `QWEN_MAX_NEW_TOKENS` (hard generation cap per chunk, default `192`)
- `QWEN_MAX_NEW_TOKENS_PER_CHAR` (dynamic output-token estimate, default `1.0`)
- `QWEN_MAX_NEW_TOKENS_BUFFER` (extra output-token headroom per chunk, default `48`)
- `QWEN_NON_STREAMING_MODE` (`0` by default; qwen's name is inverted, so `0` enables its lower-latency streaming-text path)
- `QWEN_SPEECH_TOKENIZER_DEVICE` (default `cpu`; keep vocoder/codec decoder off small GPUs)
- `QWEN_SPEECH_TOKENIZER_CPU_DTYPE` (default `float32`)
- `QWEN_SPEAKER_ENCODER_DEVICE` (default `cpu`; keep clone embedding model off small GPUs)
- `QWEN_SPEAKER_ENCODER_CPU_DTYPE` (default `float32`)
- `PYTORCH_CUDA_ALLOC_CONF` (default image/config: `expandable_segments:True`)

### GPU profiles

Small GPU / smoother buffered profile:
```bash
QWEN_SPEECH_TOKENIZER_DEVICE=cpu
QWEN_SPEAKER_ENCODER_DEVICE=cpu
QWEN_MAX_CHUNK_CHARS=220
QWEN_TRIM_CHUNK_SILENCE=1
QWEN_MAX_NEW_TOKENS=192
QWEN_MAX_NEW_TOKENS_PER_CHAR=1.0
QWEN_MAX_NEW_TOKENS_BUFFER=48
```

Larger GPU / lower-latency profile:
```bash
QWEN_SPEECH_TOKENIZER_DEVICE=cuda:0
QWEN_SPEAKER_ENCODER_DEVICE=cuda:0
QWEN_TRIM_CHUNK_SILENCE=1
QWEN_MAX_NEW_TOKENS=512
QWEN_MAX_NEW_TOKENS_PER_CHAR=2.0
QWEN_MAX_NEW_TOKENS_BUFFER=64
```

`QWEN_DEVICE` controls the main Qwen generator. `QWEN_SPEECH_TOKENIZER_DEVICE` controls the codec/vocoder decoder, and `QWEN_SPEAKER_ENCODER_DEVICE` controls clone speaker embedding extraction. Use `cpu`, `cuda:0`, or `keep`. A request with `cpu_mode=true` overrides these device settings for that request and keeps the complete pipeline on CPU.

## Notes

- TTS and ASR models share one GPU slot, are loaded on demand, and flush their generation CUDA cache after each completed request. The loaded model stays warm for `QWEN_IDLE_UNLOAD_SECONDS` (default `5` seconds) before being unloaded to release its VRAM for other GPU workloads (failed requests and `POST /reset` unload immediately). Clone speaker embeddings stay cached in CPU RAM across idle unloads and are invalidated automatically when a reference audio file changes.
- Because the container runs offline (`HF_HUB_OFFLINE=1`), pre-download `Qwen/Qwen3-ASR-0.6B` and/or `Qwen/Qwen3-ASR-1.7B` into the Hugging Face cache directory before calling `/api/asr` or `/v1/audio/transcriptions`.
- `POST /reference-audio/upload` requires `python-multipart`.
- If FlashAttention2 fails, set `QWEN_ATTN_IMPL=eager`.

## References

- Qwen3-TTS: https://github.com/QwenLM/Qwen3-TTS
- Qwen3-ASR: https://github.com/QwenLM/Qwen3-ASR
- HF TTS collection: https://huggingface.co/collections/Qwen/qwen3-tts
- HF ASR collection: https://huggingface.co/collections/Qwen/qwen3-asr
