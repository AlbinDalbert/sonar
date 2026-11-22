# Sonar

Self-hosted speech API built on [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) and
[Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR): an HTTP API that synthesizes, streams, clones,
designs voices, and transcribes speech on a GPU, plus a small web UI for TTS.

```
browser ──> ui (nginx, static Svelte build) ──/tts-api──> api (FastAPI + Qwen3-TTS, GPU)
```

The two halves are separate images and separate deployments. The API needs a GPU node and is a
multi-GB CUDA image; the UI is a few MB of nginx that can run anywhere (amd64 or arm64) and stays
up while the API restarts or reloads models.

| Directory | What | Image |
|---|---|---|
| [`api/`](api) | FastAPI service: endpoints, streaming, voices, presets, settings | `ghcr.io/albindalbert/sonar-api` (amd64) |
| [`ui/`](ui) | Svelte 5 web UI, served by nginx which proxies `/tts-api` to the API | `ghcr.io/albindalbert/sonar-ui` (amd64 + arm64) |
| [`deploy/`](deploy) | Kustomize bases for both | |

There is no authentication. Keep it on a private network.

## Quick start

```sh
# API (needs an NVIDIA GPU and the models in the Hugging Face cache; see api/README.md)
cd api && python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && pip install --no-deps qwen-tts==0.1.1
python3 -m uvicorn app.main:app --port 8888

# UI, proxying to the API on localhost:8888
cd ui && npm install && npm run dev
```

## Images

CI builds each image when its directory changes: `.github/workflows/api.yml` and `ui.yml`.
Tags: `main`, `sha-<commit>`, and `X.Y.Z` / `X.Y` for `vX.Y.Z` git tags.
