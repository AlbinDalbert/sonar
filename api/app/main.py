import base64
from collections import OrderedDict
import contextlib
import io
import itertools
import json
import os
import re
import subprocess
import tempfile
import threading
import gc
import time
import queue
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from qwen_tts import Qwen3TTSModel
from transformers import LogitsProcessor, LogitsProcessorList

try:
    from qwen_asr import Qwen3ASRModel
except ImportError:
    Qwen3ASRModel = None

SUPPORTED_LANGUAGES = [
    "Auto", "English", "Chinese", "Japanese", "Korean",
    "French", "German", "Spanish", "Italian", "Portuguese", "Arabic", "Russian",
]
ASR_SUPPORTED_LANGUAGES = [
    "Chinese", "Cantonese", "English", "Arabic", "German", "French",
    "Spanish", "Portuguese", "Indonesian", "Italian", "Korean", "Russian",
    "Thai", "Vietnamese", "Japanese", "Turkish", "Hindi", "Malay",
    "Dutch", "Swedish", "Danish", "Finnish", "Polish", "Czech",
    "Filipino", "Persian", "Greek", "Romanian", "Hungarian", "Macedonian",
]
_ASR_LANGUAGE_BY_LOWER = {lang.lower(): lang for lang in ASR_SUPPORTED_LANGUAGES}
ASR_LANGUAGE_ALIASES = {
    "zh": "Chinese",
    "yue": "Cantonese",
    "en": "English",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fil": "Filipino",
    "tl": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "ro": "Romanian",
    "hu": "Hungarian",
    "mk": "Macedonian",
}
ASR_SAMPLE_RATE = 16000
ASR_MAX_UPLOAD_MB = max(1, int(os.getenv("QWEN_ASR_MAX_UPLOAD_MB", "100")))
REFERENCE_AUDIO_DIR = Path("/reference-audio")
MODEL_IDS = {
    "clone": {
        "fast": os.getenv("QWEN_CLONE_MODEL_ID_06", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
        "quality": os.getenv("QWEN_CLONE_MODEL_ID_17", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
    },
    "design": {
        "quality": os.getenv("QWEN_DESIGN_MODEL_ID_17", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
    },
}
ASR_MODEL_IDS = {
    "fast": os.getenv("QWEN_ASR_MODEL_ID_06", "Qwen/Qwen3-ASR-0.6B"),
    "quality": os.getenv("QWEN_ASR_MODEL_ID_17", "Qwen/Qwen3-ASR-1.7B"),
}
OPENAI_AUDIO_CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}
OPENAI_VOICE_DESCRIPTIONS = {
    "alloy": "Balanced, neutral, clear mid-pitch voice with steady pacing",
    "ash": "Clear, grounded, slightly husky mid-low voice",
    "ballad": "Warm, expressive, melodic storytelling voice",
    "coral": "Warm, friendly, articulate female voice",
    "echo": "Smooth, calm, resonant male voice",
    "fable": "Expressive, articulate British-accented narrator voice",
    "onyx": "Deep, authoritative, steady male voice",
    "nova": "Bright, clear, energetic female voice",
    "sage": "Calm, thoughtful, composed voice with measured pacing",
    "shimmer": "Soft, clear, gentle female voice",
    "verse": "Dynamic, engaging, natural conversational voice",
    "marin": "Clear, natural, relaxed modern female voice",
    "cedar": "Warm, natural, grounded male voice",
}
model_status = {"state": "idle", "kind": None, "model_id": None, "device": None}  # idle | loading | ready
QWEN_MEMORY_PATCH_ENABLED = False
# Per-thread hook for frame-level streaming. While set, the patched talker
# forward hands every generated codec frame to `_frame_sink.fn` (see
# QwenService._frame_capture). Generation runs on one thread, so a
# thread-local keeps concurrent requests from seeing each other's frames.
_frame_sink = threading.local()


class StreamCancelled(Exception):
    """Raised from inside generation to stop a stream the client abandoned."""


def is_cuda_out_of_memory(exc: BaseException) -> bool:
    """Recognize CUDA allocation failures, including wrapped PyTorch errors."""
    current: Optional[BaseException] = exc
    seen: set[int] = set()
    markers = (
        "cuda out of memory",
        "cuda error: out of memory",
        "cuda_error_out_of_memory",
        "cublas_status_alloc_failed",
    )
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, torch.cuda.OutOfMemoryError):
            return True
        if any(marker in str(current).lower() for marker in markers):
            return True
        current = current.__cause__ or current.__context__
    return False


def gpu_busy_error() -> HTTPException:
    return HTTPException(status_code=503, detail="GPU busy", headers={"Retry-After": "5"})


def patch_qwen_tts_memory() -> None:
    """Reduce upstream qwen-tts generation memory use.

    qwen-tts 0.1.1 stores every transformer layer hidden state for every
    generated audio token because it uses the HF `hidden_states` channel to
    smuggle codec ids back out of `generate()`. The public API only needs the
    codec ids. Keep that channel, but replace the huge all-layer payload with
    the one last-token hidden vector the upstream wrapper expects.
    """
    global QWEN_MEMORY_PATCH_ENABLED
    try:
        from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerForConditionalGeneration
    except Exception as e:
        print(json.dumps({"event": "qwen_memory_patch_skipped", "detail": str(e)}), flush=True)
        return

    if getattr(Qwen3TTSTalkerForConditionalGeneration, "_qwen3_tts_api_memory_patch", False):
        QWEN_MEMORY_PATCH_ENABLED = True
        return

    import functools

    original_forward = Qwen3TTSTalkerForConditionalGeneration.forward

    @functools.wraps(original_forward)
    def memory_light_forward(self, *args, **kwargs):
        # GenerationMixin still needs output_hidden_states=True at the outer
        # level so it collects our custom `(last_hidden, codec_ids)` payload.
        # But the inner transformer does not need to materialize all layer
        # hidden states; that is the expensive part.
        kwargs["output_hidden_states"] = False
        output = original_forward(self, *args, **kwargs)
        codec_ids = None
        try:
            hidden_payload = getattr(output, "hidden_states", None)
            codec_ids = hidden_payload[-1] if isinstance(hidden_payload, (tuple, list)) and hidden_payload else None
            compact_last_hidden = getattr(output, "past_hidden", None)
            if compact_last_hidden is not None or codec_ids is not None:
                output.hidden_states = ((compact_last_hidden,), codec_ids)
        except Exception as e:
            print(json.dumps({"event": "qwen_memory_patch_payload_error", "detail": str(e)}), flush=True)
        # Each decode step carries the full codec frame (all code groups) for
        # the previously sampled token. Outside the try so StreamCancelled
        # propagates and aborts generate().
        sink = getattr(_frame_sink, "fn", None)
        if sink is not None and codec_ids is not None:
            sink(codec_ids)
        return output

    Qwen3TTSTalkerForConditionalGeneration.forward = memory_light_forward
    Qwen3TTSTalkerForConditionalGeneration._qwen3_tts_api_memory_patch = True
    QWEN_MEMORY_PATCH_ENABLED = True
    print(json.dumps({"event": "qwen_memory_patch_enabled"}), flush=True)


patch_qwen_tts_memory()


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to synthesize")
    language: str = Field(default="Auto", description="Language or Auto")
    audio_format: str = Field(default="wav", description="wav | ogg | pcm (pcm: /api/tts/audio-stream only)")
    mode: str = Field(default="clone", description="clone | design")
    model_size: str = Field(default="fast", description="fast | quality")
    reference_audio: Optional[str] = Field(default=None, description="Filename from /reference-audio when mode=clone")
    voice_description: Optional[str] = Field(default=None, description="Plain-text voice description when mode=design")
    voice_preset: Optional[str] = Field(default=None, description="Named voice preset to resolve into voice_description")
    user: Optional[str] = Field(default="default", description="User namespace for reference audio")
    cpu_mode: bool = Field(default=False, description="Run this request on CPU; no automatic fallback")


class OpenAISpeechRequest(BaseModel):
    model: str = Field(default="tts-1", description="OpenAI model name (tts-1, tts-1-hd, gpt-4o-mini-tts) or Sonar size/model ID")
    input: str = Field(..., min_length=1, description="Text to synthesize")
    voice: str = Field(default="alloy", description="Voice preset name, reference audio filename, OpenAI voice name, or voice description")
    response_format: str = Field(default="mp3", description="mp3 | opus | aac | flac | wav | pcm")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Playback speed (0.25 to 4.0)")
    instructions: Optional[str] = Field(default=None, description="Optional delivery instructions for design mode")
    stream_format: str = Field(default="audio", description="audio | sse")
    language: str = Field(default="Auto", description="Language or Auto")
    user: Optional[str] = Field(default="default", description="User namespace for reference audio and presets")
    cpu_mode: bool = Field(default=False, description="Run this request on CPU; no automatic fallback")


class SaveDesignRequest(BaseModel):
    filename: str = Field(..., min_length=1)
    description: Optional[str] = Field(default=None)
    audio_b64: str = Field(..., min_length=1)


class VoicePresetRequest(BaseModel):
    name: str = Field(..., min_length=1)
    voice_description: str = Field(..., min_length=1)
    description: Optional[str] = Field(default=None)


class ResetRequest(BaseModel):
    restart: bool = Field(default=False, description="Restart process after reset")


class RequestSession:
    """Exclusive model-use lease for one TTS request.

    Prevents overlapping requests from swapping or unloading each other's
    model, flushes generation CUDA cache once the total request completes,
    and hands the loaded model off to the idle-unload timer (or unloads
    immediately on error or when QWEN_IDLE_UNLOAD_SECONDS <= 0).
    """

    def __init__(self, service: "QwenService") -> None:
        self._service = service
        self._finished = False

    def finish(self, unload_now: bool = False) -> None:
        if self._finished:
            return
        self._finished = True
        self._service._finish_request(unload_now=unload_now)

    def __enter__(self) -> "RequestSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish(unload_now=exc_type is not None)


class _ManagedStream:
    """Iterator wrapper guaranteeing `on_close` runs once on end, abort, or GC."""

    def __init__(self, gen, on_close) -> None:
        self._gen = gen
        self._on_close = on_close
        self._closed = False

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._gen)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._gen.close()
        finally:
            self._on_close()

    def __del__(self) -> None:
        self.close()


class QwenService:
    def __init__(self) -> None:
        self.device = os.getenv("QWEN_DEVICE", "cuda:0")
        self.dtype = os.getenv("QWEN_DTYPE", "bfloat16")
        self.attn_impl = os.getenv("QWEN_ATTN_IMPL", "flash_attention_2")
        self.cpu_dtype = os.getenv("QWEN_CPU_DTYPE", "float32")
        self.cpu_attn_impl = os.getenv("QWEN_CPU_ATTN_IMPL", "sdpa")
        # Force local-only model resolution by default so runtime never depends on
        # live Hugging Face API checks (works with pre-populated HF cache).
        self.local_files_only = os.getenv("QWEN_LOCAL_FILES_ONLY", "1").lower() not in {"0", "false", "no"}
        self.max_new_tokens = int(os.getenv("QWEN_MAX_NEW_TOKENS", "192"))
        self.max_new_tokens_per_char = float(os.getenv("QWEN_MAX_NEW_TOKENS_PER_CHAR", "1.0"))
        self.max_new_tokens_buffer = int(os.getenv("QWEN_MAX_NEW_TOKENS_BUFFER", "48"))
        self.asr_max_new_tokens = max(64, int(os.getenv("QWEN_ASR_MAX_NEW_TOKENS", "2048")))
        self.asr_max_inference_batch_size = max(1, int(os.getenv("QWEN_ASR_MAX_INFERENCE_BATCH_SIZE", "1")))
        # Keep the model warm briefly for back-to-back requests, then unload so
        # other GPU workloads get VRAM back quickly. Set 0 to unload immediately.
        self.idle_unload_seconds = max(0.0, float(os.getenv("QWEN_IDLE_UNLOAD_SECONDS", "5")))
        self.speaker_cache_max_items = max(0, int(os.getenv("QWEN_SPEAKER_CACHE_MAX_ITEMS", "64")))
        # On ~3 GiB GPUs the 0.6B talker can fit, but the auxiliary vocoder /
        # speech tokenizer and speaker encoder leave almost no generation
        # headroom. Keep those aux modules on CPU by default; only the talker
        # stays on CUDA for generation.
        self.speech_tokenizer_device = os.getenv("QWEN_SPEECH_TOKENIZER_DEVICE", "cpu").strip().lower()
        self.speech_tokenizer_cpu_dtype = os.getenv("QWEN_SPEECH_TOKENIZER_CPU_DTYPE", "float32").strip().lower()
        self.speaker_encoder_device = os.getenv("QWEN_SPEAKER_ENCODER_DEVICE", "cpu").strip().lower()
        self.speaker_encoder_cpu_dtype = os.getenv("QWEN_SPEAKER_ENCODER_CPU_DTYPE", "float32").strip().lower()
        # qwen's parameter is inverted: non_streaming_mode=False enables its
        # lower-latency streaming-text path. Keep false by default.
        self.non_streaming_mode = os.getenv("QWEN_NON_STREAMING_MODE", "0").lower() in {"1", "true", "yes"}
        self.request_lock = threading.Lock()
        self.lock = threading.Lock()
        self.generation_lock = threading.Lock()
        self._model = None
        self._model_kind: Optional[str] = None
        self._model_id: Optional[str] = None
        self._model_device: Optional[str] = None
        self._unload_timer: Optional[threading.Timer] = None
        self._idle_generation: int = 0
        self._speaker_cache: OrderedDict[tuple, torch.Tensor] = OrderedDict()
        self._speaker_cache_lock = threading.Lock()

    def _torch_dtype(self, device: str):
        dtype_name = self.cpu_dtype if device == "cpu" else self.dtype
        dtype = self._dtype_from_name(dtype_name)
        if dtype is None:
            raise ValueError(f"QWEN_{'CPU_' if device == 'cpu' else ''}DTYPE must specify a dtype")
        return dtype

    def _dtype_from_name(self, name: str):
        normalized = (name or "").strip().lower()
        if normalized in {"", "none", "keep"}:
            return None
        if normalized in {"float32", "fp32"}:
            return torch.float32
        if normalized in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if normalized in {"float16", "fp16", "half"}:
            return torch.float16
        raise ValueError(f"unsupported dtype: {name}")

    def _move_module(self, module, device_name: str, cpu_dtype_name: str, label: str) -> None:
        device_name = (device_name or "").strip().lower()
        if device_name in {"", "none", "keep"} or module is None:
            return

        dtype = self._dtype_from_name(cpu_dtype_name) if device_name == "cpu" else None
        try:
            if dtype is None:
                module.to(device_name)
            else:
                module.to(device=device_name, dtype=dtype)
            print(json.dumps({
                "event": "tts_aux_module_moved",
                "module": label,
                "device": device_name,
                "dtype": str(dtype) if dtype is not None else "keep",
                "cuda": cuda_memory_snapshot(),
            }), flush=True)
        except Exception as e:
            print(json.dumps({
                "event": "tts_aux_module_move_failed",
                "module": label,
                "device": device_name,
                "detail": str(e),
            }), flush=True)
            raise

    def _apply_post_load_device_policy(self, wrapped_model, model_device: str) -> None:
        qwen_model = getattr(wrapped_model, "model", None)
        speech_tokenizer = getattr(qwen_model, "speech_tokenizer", None)
        speech_tokenizer_model = getattr(speech_tokenizer, "model", None)
        # cpu_mode applies to the complete pipeline. It must never move an
        # auxiliary module back to CUDA through the normal GPU profile.
        speech_tokenizer_device = "cpu" if model_device == "cpu" else self.speech_tokenizer_device
        speaker_encoder_device = "cpu" if model_device == "cpu" else self.speaker_encoder_device
        self._move_module(
            speech_tokenizer_model,
            speech_tokenizer_device,
            self.speech_tokenizer_cpu_dtype,
            "speech_tokenizer",
        )
        if speech_tokenizer is not None and speech_tokenizer_device not in {"", "none", "keep"}:
            speech_tokenizer.device = torch.device(speech_tokenizer_device)

        self._move_module(
            getattr(qwen_model, "speaker_encoder", None),
            speaker_encoder_device,
            self.speaker_encoder_cpu_dtype,
            "speaker_encoder",
        )
        cleanup_generation_memory()

    def _cancel_idle_unload_locked(self) -> None:
        self._idle_generation += 1
        if self._unload_timer is not None:
            self._unload_timer.cancel()
            self._unload_timer = None

    def _schedule_idle_unload_locked(self) -> None:
        self._cancel_idle_unload_locked()
        if self.idle_unload_seconds <= 0 or self._model is None:
            return
        expected_gen = self._idle_generation
        timer = threading.Timer(self.idle_unload_seconds, self._on_idle_unload_timeout, args=(expected_gen,))
        timer.daemon = True
        self._unload_timer = timer
        timer.start()

    def _on_idle_unload_timeout(self, expected_gen: int) -> None:
        if not self.request_lock.acquire(blocking=False):
            return
        unloaded_model_id = None
        had_model = False
        try:
            with self.lock:
                if self._idle_generation != expected_gen:
                    return
                self._unload_timer = None
                unloaded_model_id = self._model_id
                had_model = self._model is not None
                self._unload_model_locked()
                model_status.update({"state": "idle", "kind": None, "model_id": None, "device": None})
        finally:
            self.request_lock.release()

        if had_model:
            print(json.dumps({
                "event": "tts_model_unloaded",
                "reason": "idle",
                "model_id": unloaded_model_id,
                "cuda": cuda_memory_snapshot(),
            }), flush=True)

    def _unload_model_locked(self):
        if self._model is not None:
            del self._model
            self._model = None
        self._model_kind = None
        self._model_id = None
        self._model_device = None
        cleanup_generation_memory()

    def begin_request(self) -> RequestSession:
        """Acquire exclusive model access for one TTS or ASR request."""
        self.request_lock.acquire()
        with self.lock:
            self._cancel_idle_unload_locked()
        return RequestSession(self)

    def request_session(self) -> RequestSession:
        return self.begin_request()

    def _finish_request(self, unload_now: bool = False) -> None:
        unloaded_model_id = None
        had_model = False
        try:
            if unload_now or self.idle_unload_seconds <= 0:
                with self.lock:
                    self._cancel_idle_unload_locked()
                    unloaded_model_id = self._model_id
                    had_model = self._model is not None
                    self._unload_model_locked()
                    model_status.update({"state": "idle", "kind": None, "model_id": None, "device": None})
            else:
                # Release intermediate generation buffers right away while
                # keeping the loaded model warm until the idle timer fires.
                cleanup_generation_memory()
                with self.lock:
                    if self._model is not None:
                        self._schedule_idle_unload_locked()
        finally:
            self.request_lock.release()

        if had_model:
            print(json.dumps({
                "event": "tts_model_unloaded",
                "reason": "error" if unload_now else "immediate",
                "model_id": unloaded_model_id,
                "cuda": cuda_memory_snapshot(),
            }), flush=True)

    def release_idle_cuda_memory(self, min_mb: int) -> None:
        if self.loaded_model() is not None:
            return
        if torch.cuda.memory_reserved() < min_mb * 1024 * 1024:
            return
        if not self.request_lock.acquire(blocking=False):
            return
        try:
            if self.loaded_model() is not None:
                return
            before_mb = int(torch.cuda.memory_reserved() / (1024 * 1024))
            cleanup_generation_memory()
        finally:
            self.request_lock.release()
        print(json.dumps({
            "event": "tts_idle_cuda_released",
            "reserved_before_mb": before_mb,
            "cuda": cuda_memory_snapshot(),
        }), flush=True)

    def get_model(self, model_id: str, cpu_mode: bool = False, kind: str = "tts"):
        model_device = "cpu" if cpu_mode else self.device.strip().lower()
        with self.lock:
            if (
                self._model is not None
                and self._model_kind == kind
                and self._model_id == model_id
                and self._model_device == model_device
            ):
                return self._model

            if kind == "asr" and Qwen3ASRModel is None:
                raise HTTPException(status_code=501, detail="ASR unavailable: qwen-asr package is not installed")

            model_status.update({"state": "loading", "kind": kind, "model_id": model_id, "device": model_device})
            self._unload_model_locked()

            try:
                if kind == "asr":
                    self._model = Qwen3ASRModel.from_pretrained(
                        model_id,
                        device_map=model_device,
                        dtype=self._torch_dtype(model_device),
                        attn_implementation=self.cpu_attn_impl if model_device == "cpu" else self.attn_impl,
                        local_files_only=self.local_files_only,
                        max_inference_batch_size=self.asr_max_inference_batch_size,
                        max_new_tokens=self.asr_max_new_tokens,
                    )
                else:
                    self._model = Qwen3TTSModel.from_pretrained(
                        model_id,
                        device_map=model_device,
                        dtype=self._torch_dtype(model_device),
                        attn_implementation=self.cpu_attn_impl if model_device == "cpu" else self.attn_impl,
                        local_files_only=self.local_files_only,
                    )
                    self._apply_post_load_device_policy(self._model, model_device)
            except Exception as e:
                # A failed load can leave a partially constructed model and
                # CUDA cache allocations behind. Release both before replying.
                self._unload_model_locked()
                model_status.update({"state": "idle", "kind": None, "model_id": None, "device": None})
                if model_device.startswith("cuda") and is_cuda_out_of_memory(e):
                    print(json.dumps({
                        "event": f"{kind}_model_load_gpu_busy",
                        "model_id": model_id,
                        "device": model_device,
                        "detail": str(e),
                        "cuda": cuda_memory_snapshot(),
                    }), flush=True)
                    raise gpu_busy_error() from None
                raise

            self._model_kind = kind
            self._model_id = model_id
            self._model_device = model_device
            model_status.update({"state": "ready", "kind": kind, "model_id": model_id, "device": model_device})
            return self._model

    def unload_model(self, reason: str = "manual") -> None:
        """Drop the loaded model and release its CUDA allocations."""
        with self.request_lock:
            with self.generation_lock:
                with self.lock:
                    self._cancel_idle_unload_locked()
                    unloaded_model_id = self._model_id
                    had_model = self._model is not None
                    self._unload_model_locked()
                    model_status.update({"state": "idle", "kind": None, "model_id": None, "device": None})

        if had_model:
            print(json.dumps({
                "event": "tts_model_unloaded",
                "reason": reason,
                "model_id": unloaded_model_id,
                "cuda": cuda_memory_snapshot(),
            }), flush=True)

    def _get_cached_speaker_embedding(self, key: tuple) -> Optional[torch.Tensor]:
        if self.speaker_cache_max_items <= 0:
            return None
        with self._speaker_cache_lock:
            emb = self._speaker_cache.get(key)
            if emb is None:
                return None
            self._speaker_cache.move_to_end(key)
            return emb.clone()

    def _put_cached_speaker_embedding(self, key: tuple, emb: torch.Tensor) -> None:
        if self.speaker_cache_max_items <= 0:
            return
        with self._speaker_cache_lock:
            self._speaker_cache[key] = emb.detach().cpu().clone()
            self._speaker_cache.move_to_end(key)
            while len(self._speaker_cache) > self.speaker_cache_max_items:
                self._speaker_cache.popitem(last=False)

    def clear_speaker_embedding_cache(self) -> None:
        with self._speaker_cache_lock:
            self._speaker_cache.clear()

    def hard_reset(self):
        self.clear_speaker_embedding_cache()
        self.unload_model(reason="reset")

    def resolve_reference_audio(self, filename: str, user_dir: Path) -> Path:
        candidate = (user_dir / filename).resolve()
        if not str(candidate).startswith(str(user_dir.resolve())):
            raise HTTPException(status_code=400, detail="Invalid filename")
        if not candidate.is_file():
            raise HTTPException(status_code=404, detail=f"Reference audio not found: {filename}")
        return candidate

    def validate_request(self, req: TTSRequest, user_dir: Path) -> None:
        """Validate request fields before waiting for the GPU lock."""
        mode = (req.mode or "clone").strip().lower()
        if mode == "clone":
            if not req.reference_audio:
                raise HTTPException(status_code=400, detail="reference_audio required for clone mode")
            self.resolve_reference_audio(req.reference_audio, user_dir)
        elif mode == "design":
            if not req.voice_description:
                raise HTTPException(status_code=400, detail="voice_description required for design mode")
        else:
            raise HTTPException(status_code=400, detail="mode must be clone or design")

    def _resolve_model_size(self, mode: str, requested_size: Optional[str]) -> str:
        size = (requested_size or "fast").strip().lower()
        if size not in {"fast", "quality"}:
            size = "fast"

        if mode == "design" and size == "fast":
            size = "quality"

        return size

    def _max_new_tokens_for_text(self, text: str) -> int:
        estimated = int(len(text or "") * self.max_new_tokens_per_char) + self.max_new_tokens_buffer
        return max(32, min(self.max_new_tokens, estimated))

    def _raise_gpu_busy_if_oom(self, exc: BaseException) -> None:
        if (self._model_device or "").startswith("cuda") and is_cuda_out_of_memory(exc):
            cleanup_generation_memory()
            raise gpu_busy_error() from None

    def create_xvector_voice_clone_prompt(self, req: TTSRequest, user_dir: Path):
        """Create a lightweight voice-clone prompt once per request.

        Upstream qwen-tts currently runs the speech tokenizer on the reference
        audio even when x_vector_only_mode=True, then discards those ref codes.
        For our clone mode we only use the speaker embedding, so skip that
        extra encode path and cache extracted embeddings by model and file stat.
        """
        mode = (req.mode or "clone").strip().lower()
        if mode != "clone":
            return None
        if not req.reference_audio:
            raise HTTPException(status_code=400, detail="reference_audio required for clone mode")

        size = self._resolve_model_size(mode, req.model_size)
        if size not in MODEL_IDS[mode]:
            size = "fast"
        model_id = MODEL_IDS[mode][size]
        model = self.get_model(model_id, req.cpu_mode)
        ref_path = self.resolve_reference_audio(req.reference_audio, user_dir)

        speaker_encoder = model.model.speaker_encoder
        if speaker_encoder is None:
            raise HTTPException(
                status_code=500,
                detail="failed to create voice clone prompt: loaded model has no speaker_encoder",
            )
        try:
            spk_param = next(speaker_encoder.parameters())
            spk_device = spk_param.device
            spk_dtype = spk_param.dtype
        except StopIteration:
            spk_device = torch.device("cpu")
            spk_dtype = torch.float32

        try:
            ref_stat = ref_path.stat()
            cache_key = (model_id, str(ref_path), ref_stat.st_mtime_ns, ref_stat.st_size, str(spk_dtype))
        except OSError:
            cache_key = None

        if cache_key is not None:
            cached_emb = self._get_cached_speaker_embedding(cache_key)
            if cached_emb is not None:
                return {
                    "ref_code": [None],
                    "ref_spk_embedding": [cached_emb],
                    "x_vector_only_mode": [True],
                    "icl_mode": [False],
                }

        with self.generation_lock:
            try:
                import librosa

                normalized = model._normalize_audio_inputs([str(ref_path)])
                wav, sr = normalized[0]
                wav_resample = wav
                target_sr = model.model.speaker_encoder_sample_rate
                if sr != target_sr:
                    wav_resample = librosa.resample(
                        y=wav_resample.astype(np.float32),
                        orig_sr=int(sr),
                        target_sr=int(target_sr),
                    )
                from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

                mels = mel_spectrogram(
                    torch.from_numpy(wav_resample.astype(np.float32)).unsqueeze(0),
                    n_fft=1024,
                    num_mels=128,
                    sampling_rate=24000,
                    hop_size=256,
                    win_size=1024,
                    fmin=0,
                    fmax=12000,
                ).transpose(1, 2)
                with torch.inference_mode():
                    spk_emb = speaker_encoder(mels.to(spk_device).to(spk_dtype))[0].detach().cpu()
                if cache_key is not None:
                    self._put_cached_speaker_embedding(cache_key, spk_emb)
                return {
                    "ref_code": [None],
                    "ref_spk_embedding": [spk_emb],
                    "x_vector_only_mode": [True],
                    "icl_mode": [False],
                }
            except Exception as e:
                cleanup_generation_memory()
                self._raise_gpu_busy_if_oom(e)
                raise HTTPException(status_code=500, detail=f"failed to create voice clone prompt: {e}")

    def prepare_request_extra_kwargs(self, req: TTSRequest, user_dir: Path) -> dict:
        """Validate and load before an endpoint commits streaming headers."""
        mode = (req.mode or "clone").strip().lower()
        if mode == "clone":
            prompt = self.create_xvector_voice_clone_prompt(req, user_dir)
            return {"voice_clone_prompt": prompt}
        if mode != "design":
            raise HTTPException(status_code=400, detail="mode must be clone or design")
        if not req.voice_description:
            raise HTTPException(status_code=400, detail="voice_description required for design mode")

        size = self._resolve_model_size(mode, req.model_size)
        if size not in MODEL_IDS[mode]:
            size = "quality"
        self.get_model(MODEL_IDS[mode][size], req.cpu_mode)
        return {}

    def loaded_model(self):
        return self._model

    @contextlib.contextmanager
    def _frame_capture(self, model, on_frame):
        """Stream codec frames out of generation and skip the final decode.

        Must be entered while holding generation_lock: it temporarily shadows
        the speech tokenizer's decode() on the instance, because the caller
        decodes the frames incrementally itself.
        """
        if on_frame is None:
            yield
            return
        tokenizer = model.model.speech_tokenizer
        sample_rate = int(tokenizer.get_output_sample_rate())
        eos_id = int(model.model.config.talker_config.codec_eos_token_id)
        finished = False

        def sink(codec_ids):
            nonlocal finished
            if finished:
                return
            frame = codec_ids[0].detach().to("cpu")
            if int(frame[0]) == eos_id:
                finished = True
                return
            on_frame(frame)

        def skip_decode(encoded):
            n = len(encoded) if isinstance(encoded, list) else 1
            return [np.zeros(0, dtype=np.float32) for _ in range(n)], sample_rate

        tokenizer.decode = skip_decode
        _frame_sink.fn = sink
        try:
            yield
        finally:
            _frame_sink.fn = None
            tokenizer.__dict__.pop("decode", None)

    def _resolve_asr_model_size(self, requested_size: Optional[str]) -> str:
        size = (requested_size or "fast").strip().lower()
        if size not in ASR_MODEL_IDS:
            return "fast"
        return size

    def transcribe(
        self,
        wav: np.ndarray,
        sample_rate: int,
        model_size: Optional[str] = "fast",
        language: Optional[str] = None,
        context: str = "",
        cpu_mode: bool = False,
    ) -> tuple[str, str, str, str]:
        """Transcribe a mono float32 audio array using Qwen3-ASR."""
        size = self._resolve_asr_model_size(model_size)
        model_id = ASR_MODEL_IDS[size]
        model = self.get_model(model_id, cpu_mode=cpu_mode, kind="asr")
        print(json.dumps({
            "event": "asr_transcribe_start",
            "model_id": model_id,
            "model_size": size,
            "duration_s": round(float(wav.size) / float(sample_rate), 3),
            "language": language or "Auto",
            "cuda": cuda_memory_snapshot(),
        }), flush=True)
        try:
            with self.generation_lock:
                results = model.transcribe(
                    audio=(wav, sample_rate),
                    context=context or "",
                    language=language,
                )
        except Exception as e:
            self._raise_gpu_busy_if_oom(e)
            raise

        if not results:
            return "", language or "", model_id, size
        first = results[0]
        return (getattr(first, "text", "") or "").strip(), (getattr(first, "language", "") or ""), model_id, size

    def synthesize(self, req: TTSRequest, user_dir: Path, extra_kwargs: Optional[dict] = None, on_frame=None):
        """Generate one text chunk.

        With `on_frame`, each codec frame (CPU LongTensor of code groups) is
        passed to it as it is generated and the returned wav is empty; the
        caller decodes the frames (see StreamingCodecDecoder).
        """
        extra_kwargs = extra_kwargs.copy() if extra_kwargs else {}
        extra_kwargs.setdefault("max_new_tokens", self._max_new_tokens_for_text(req.text))
        extra_kwargs.setdefault("non_streaming_mode", self.non_streaming_mode)
        print(json.dumps({
            "event": "tts_generate_start",
            "chars": len(req.text or ""),
            "max_new_tokens": extra_kwargs.get("max_new_tokens"),
            "non_streaming_mode": extra_kwargs.get("non_streaming_mode"),
            "cuda": cuda_memory_snapshot(),
        }), flush=True)
        mode = (req.mode or "clone").strip().lower()
        if mode not in MODEL_IDS:
            raise HTTPException(status_code=400, detail="mode must be clone or design")

        size = self._resolve_model_size(mode, req.model_size)
        if size not in MODEL_IDS[mode]:
            size = "quality" if mode == "design" else "fast"

        model_id = MODEL_IDS[mode][size]

        if mode == "clone":
            if not req.reference_audio:
                raise HTTPException(status_code=400, detail="reference_audio required for clone mode")
            model = self.get_model(model_id, req.cpu_mode)
            ref_path = self.resolve_reference_audio(req.reference_audio, user_dir)
            try:
                with self.generation_lock, self._frame_capture(model, on_frame):
                    wavs, sr = model.generate_voice_clone(
                        text=req.text,
                        language=req.language or "Auto",
                        ref_audio=str(ref_path),
                        x_vector_only_mode=True,
                        **extra_kwargs,
                    )
            except Exception as e:
                self._raise_gpu_busy_if_oom(e)
                raise
        else:
            if not req.voice_description:
                raise HTTPException(status_code=400, detail="voice_description required for design mode")
            model = self.get_model(model_id, req.cpu_mode)
            try:
                with self.generation_lock, self._frame_capture(model, on_frame):
                    wavs, sr = model.generate_voice_design(
                        text=req.text,
                        language=req.language or "Auto",
                        instruct=req.voice_description,
                        **extra_kwargs,
                    )
            except Exception as e:
                self._raise_gpu_busy_if_oom(e)
                raise

        return wavs[0], sr, model_id, size


app = FastAPI(title="qwen3-tts-api", version="0.7.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("QWEN_CORS_ORIGINS", "*").split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
    # Lets browser/Obsidian fetch() clients read the PCM stream format.
    expose_headers=["X-Sample-Rate", "X-Channels", "X-Sample-Format"],
)
svc = QwenService()
start_time = time.time()


def cuda_memory_snapshot():
    if not torch.cuda.is_available():
        return None
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return {
            "free_mb": int(free_bytes / (1024 * 1024)),
            "total_mb": int(total_bytes / (1024 * 1024)),
            "allocated_mb": int(torch.cuda.memory_allocated() / (1024 * 1024)),
            "reserved_mb": int(torch.cuda.memory_reserved() / (1024 * 1024)),
            "max_allocated_mb": int(torch.cuda.max_memory_allocated() / (1024 * 1024)),
        }
    except Exception as e:
        return {"error": str(e)}


def cleanup_generation_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Text normalization + chunking
# ---------------------------------------------------------------------------
MAX_CHUNK_CHARS = int(os.getenv("QWEN_MAX_CHUNK_CHARS", "160"))
TRIM_CHUNK_SILENCE = os.getenv("QWEN_TRIM_CHUNK_SILENCE", "1").lower() not in {"0", "false", "no"}
TRIM_SILENCE_THRESHOLD = float(os.getenv("QWEN_TRIM_SILENCE_THRESHOLD", "0.004"))
TRIM_SILENCE_RELATIVE_THRESHOLD = float(os.getenv("QWEN_TRIM_SILENCE_RELATIVE_THRESHOLD", "0.02"))
TRIM_SILENCE_KEEP_MS = int(os.getenv("QWEN_TRIM_SILENCE_KEEP_MS", "60"))
CHUNK_FADE_MS = int(os.getenv("QWEN_CHUNK_FADE_MS", "8"))
# Frame-level streaming (/api/tts/audio-stream). The 12Hz codec emits one
# frame per 80 ms of audio. The first decode waits for FIRST_FRAMES; after
# that the stream decodes whatever has accumulated (at least STEP_FRAMES).
STREAM_FIRST_FRAMES = max(1, int(os.getenv("QWEN_STREAM_FIRST_FRAMES", "4")))
STREAM_STEP_FRAMES = max(1, int(os.getenv("QWEN_STREAM_STEP_FRAMES", "4")))
STREAM_WAV_CHANNELS = 1
STREAM_WAV_BITS_PER_SAMPLE = 16


def wav_stream_header(sample_rate: int) -> bytes:
    """Return a WAV header suitable for HTTP chunked streaming.

    The final data length is unknown, so sizes are set to 0xffffffff. Most
    browsers/players accept this for live-ish WAV streams.
    """
    byte_rate = sample_rate * STREAM_WAV_CHANNELS * STREAM_WAV_BITS_PER_SAMPLE // 8
    block_align = STREAM_WAV_CHANNELS * STREAM_WAV_BITS_PER_SAMPLE // 8
    return (
        b"RIFF" + (0xFFFFFFFF).to_bytes(4, "little") + b"WAVE"
        + b"fmt " + (16).to_bytes(4, "little")
        + (1).to_bytes(2, "little")
        + STREAM_WAV_CHANNELS.to_bytes(2, "little")
        + int(sample_rate).to_bytes(4, "little")
        + int(byte_rate).to_bytes(4, "little")
        + int(block_align).to_bytes(2, "little")
        + STREAM_WAV_BITS_PER_SAMPLE.to_bytes(2, "little")
        + b"data" + (0xFFFFFFFF).to_bytes(4, "little")
    )


def audio_to_float_mono(wav) -> np.ndarray:
    data = np.asarray(wav, dtype=np.float32)
    if data.ndim > 1:
        data = data.reshape(-1)
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)


def postprocess_chunk_audio(wav, sample_rate: int) -> np.ndarray:
    data = audio_to_float_mono(wav)
    if not TRIM_CHUNK_SILENCE or data.size == 0:
        return data

    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak <= 0.0:
        return data

    threshold = max(TRIM_SILENCE_THRESHOLD, peak * TRIM_SILENCE_RELATIVE_THRESHOLD)
    voiced = np.flatnonzero(np.abs(data) > threshold)
    if voiced.size == 0:
        return data

    keep = int(sample_rate * TRIM_SILENCE_KEEP_MS / 1000)
    start = max(0, int(voiced[0]) - keep)
    end = min(data.size, int(voiced[-1]) + keep)
    trimmed = data[start:end].copy()

    fade = min(int(sample_rate * CHUNK_FADE_MS / 1000), trimmed.size // 2)
    if fade > 1:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        trimmed[:fade] *= ramp
        trimmed[-fade:] *= ramp[::-1]

    return trimmed


def float_wav_to_pcm16_bytes(wav) -> bytes:
    data = audio_to_float_mono(wav)
    data = np.clip(data, -1.0, 1.0)
    return (data * 32767.0).astype("<i2", copy=False).tobytes()


class StreamingCodecDecoder:
    """Stateful, exact incremental decoding of 12Hz codec frames.

    Mirrors Qwen3TTSTokenizerV2Decoder.forward (qwen-tts 0.1.1) but carries
    state between calls instead of re-decoding context: each causal conv keeps
    its last `padding` input samples, each transposed conv keeps the overlap it
    still owes the next span, and the sliding-window transformer keeps a KV
    cache. Pointwise modules (quantizer lookup + 1x1 projections, SnakeBeta,
    norms, linears) need no state. Every frame is decoded exactly once and the
    output matches a one-shot decode of the whole sequence.

    One instance per text chunk (the talker restarts per chunk too).
    """

    def __init__(self, speech_tokenizer):
        from transformers.cache_utils import DynamicCache
        from qwen_tts.core.tokenizer_12hz import modeling_qwen3_tts_tokenizer_v2 as v2

        self.v2 = v2
        self.decoder = speech_tokenizer.model.decoder
        self.device = speech_tokenizer.device
        self.upsample = int(self.decoder.total_upsample)
        self.kv_cache = DynamicCache(config=self.decoder.pre_transformer.config)
        self.state: dict[int, torch.Tensor] = {}
        self.frames: list[torch.Tensor] = []
        self.started = False

    def push(self, frame: torch.Tensor) -> None:
        self.frames.append(frame)

    def ready(self) -> bool:
        return len(self.frames) >= (STREAM_STEP_FRAMES if self.started else STREAM_FIRST_FRAMES)

    def decode_pending(self) -> np.ndarray:
        if not self.frames:
            return np.zeros(0, dtype=np.float32)
        codes = torch.stack(self.frames, dim=0)  # (T, Q)
        self.frames = []
        codes = codes.clamp(min=0).transpose(0, 1).unsqueeze(0).to(self.device)  # (1, Q, T)
        with torch.inference_mode():
            wav = self._forward(codes)
        self.started = True
        return wav.reshape(-1).float().cpu().numpy()

    # -- stateful mirrors of the upstream modules ---------------------------

    def _forward(self, codes: torch.Tensor) -> torch.Tensor:
        dec = self.decoder
        hidden = dec.quantizer.decode(codes)
        hidden = self._run(dec.pre_conv, hidden).transpose(1, 2)
        hidden = dec.pre_transformer(
            inputs_embeds=hidden, past_key_values=self.kv_cache, use_cache=True
        ).last_hidden_state
        hidden = hidden.permute(0, 2, 1)
        for blocks in dec.upsample:
            for block in blocks:
                hidden = self._run(block, hidden)
        for block in dec.decoder:
            hidden = self._run(block, hidden)
        return hidden.clamp(min=-1, max=1)

    def _run(self, m, x: torch.Tensor) -> torch.Tensor:
        v2 = self.v2
        if isinstance(m, v2.Qwen3TTSTokenizerV2CausalConvNet):
            return self._causal_conv(m, x)
        if isinstance(m, v2.Qwen3TTSTokenizerV2CausalTransConvNet):
            return self._causal_trans_conv(m, x)
        if isinstance(m, v2.Qwen3TTSTokenizerV2ConvNeXtBlock):
            h = self._causal_conv(m.dwconv, x).permute(0, 2, 1)
            h = m.pwconv2(m.act(m.pwconv1(m.norm(h))))
            return x + (m.gamma * h).permute(0, 2, 1)
        if isinstance(m, v2.Qwen3TTSTokenizerV2DecoderDecoderResidualUnit):
            h = self._causal_conv(m.conv1, m.act1(x))
            return x + self._causal_conv(m.conv2, m.act2(h))
        if isinstance(m, v2.Qwen3TTSTokenizerV2DecoderDecoderBlock):
            for sub in m.block:
                x = self._run(sub, x)
            return x
        if isinstance(m, v2.SnakeBeta):
            return m(x)
        raise RuntimeError(f"streaming decoder: unsupported module {type(m).__name__}")

    def _causal_conv(self, m, x: torch.Tensor) -> torch.Tensor:
        # Upstream left-pads with `padding` zeros; carry the real samples instead.
        if m.stride != 1:
            raise RuntimeError("streaming decoder: strided causal conv not supported")
        prev = self.state.get(id(m))
        if prev is None:
            prev = x.new_zeros(x.shape[0], x.shape[1], m.padding)
        x = torch.cat([prev, x], dim=-1)
        self.state[id(m)] = x[..., x.shape[-1] - m.padding:].clone()
        return m.conv(x)

    def _causal_trans_conv(self, m, x: torch.Tensor) -> torch.Tensor:
        # Each input step writes `kernel` samples, `overlap` of them into the
        # next step's span. Hold that tail back until the next call adds to it.
        conv = m.conv
        stride = conv.stride[0]
        overlap = conv.kernel_size[0] - stride
        y = F.conv_transpose1d(x, conv.weight, None, stride=stride)
        prev = self.state.get(id(m))
        if prev is not None:
            y[..., :overlap] += prev
        if overlap:
            self.state[id(m)] = y[..., -overlap:].clone()
            y = y[..., :-overlap]
        if conv.bias is not None:
            y = y + conv.bias.view(1, -1, 1)
        return y


class ChunkEdgeShaper:
    """Streaming version of postprocess_chunk_audio for one text chunk.

    Drops leading silence (keeping TRIM_SILENCE_KEEP_MS) before the first
    voiced sample and fades both edges. Audio after the last voiced sample is
    held back: a pause inside speech is released once voice resumes, and
    whatever is still held when the chunk ends is trailing silence and gets
    trimmed. Voiced audio is never delayed (beyond the fade length).
    """

    def __init__(self, sample_rate: int):
        self.keep = int(sample_rate * TRIM_SILENCE_KEEP_MS / 1000)
        self.fade = int(sample_rate * CHUNK_FADE_MS / 1000)
        self.tail = np.zeros(0, dtype=np.float32)
        self.peak = 0.0
        self.voiced = False

    def _threshold(self) -> float:
        return max(TRIM_SILENCE_THRESHOLD, self.peak * TRIM_SILENCE_RELATIVE_THRESHOLD)

    def feed(self, piece: np.ndarray) -> np.ndarray:
        piece = audio_to_float_mono(piece)
        if not TRIM_CHUNK_SILENCE:
            return piece
        if piece.size:
            self.peak = max(self.peak, float(np.max(np.abs(piece))))
        data = np.concatenate([self.tail, piece])
        if not self.voiced:
            voiced = np.flatnonzero(np.abs(data) > self._threshold())
            if voiced.size == 0:
                self.tail = data[-self.keep:] if self.keep else data[:0]
                return data[:0]
            data = data[max(0, int(voiced[0]) - self.keep):].copy()
            fade = min(self.fade, data.size)
            if fade > 1:
                data[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
            self.voiced = True
        loud = np.flatnonzero(np.abs(data) > self._threshold())
        # Always keep `fade` samples back so finish() can fade out the end.
        cut = min(int(loud[-1]) + 1 if loud.size else 0, max(0, data.size - self.fade))
        self.tail = data[cut:]
        return data[:cut]

    def finish(self) -> np.ndarray:
        if not TRIM_CHUNK_SILENCE:
            return np.zeros(0, dtype=np.float32)
        data = self.tail
        self.tail = data[:0]
        if not self.voiced or data.size == 0:
            return data[:0]
        voiced = np.flatnonzero(np.abs(data) > self._threshold())
        end = int(voiced[-1]) + self.keep if voiced.size else self.keep
        data = data[:min(end, data.size)].copy()
        fade = min(self.fade, data.size)
        if fade > 1:
            data[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        return data


def normalize_text(text: str) -> str:
    """Clean up messy whitespace before synthesis."""
    # Collapse 3+ newlines → paragraph break
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces/tabs (but not newlines) → single space
    text = re.sub(r"[^\S\n]+", " ", text)
    # Strip trailing spaces on each line
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


def _group_parts(parts: list[str], max_chars: int, sep: str = " ") -> list[str]:
    """Greedily join parts into chunks up to max_chars."""
    result: list[str] = []
    current = ""
    for part in parts:
        if not current:
            current = part
        elif len(current) + len(sep) + len(part) <= max_chars:
            current += sep + part
        else:
            result.append(current)
            current = part
    if current:
        result.append(current)
    return result


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split text into contextually-aware chunks.

    Strategy (in order):
      1. Normalize whitespace.
      2. If the whole text fits, return as-is.
      3. Split on paragraph breaks (double newline).
      4. If a paragraph is still too long, split on single newlines first
         (preferred — natural line breaks mid-paragraph).
      5. If a line-group is still too long, fall back to sentence boundaries.
    """
    text = normalize_text(text)

    if len(text) <= max_chars:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks: list[str] = []
    for para in paragraphs:
        if len(para) <= max_chars:
            chunks.append(para)
            continue

        # Prefer splitting on single newlines (explicit line breaks)
        lines = [l.strip() for l in para.split("\n") if l.strip()]
        line_groups = _group_parts(lines, max_chars)

        for group in line_groups:
            if len(group) <= max_chars:
                chunks.append(group)
            else:
                # Fall back to sentence boundaries
                sentences = re.split(r"(?<=[.!?])\s+", group)
                chunks.extend(_group_parts(sentences, max_chars))

    return chunks if chunks else [text]


def meta_path_for(directory: Path, filename: str) -> Path:
    return directory / f"{filename}.meta.json"


def write_meta(directory: Path, filename: str, source: str, description: Optional[str] = None):
    payload = {"source": source}
    if description is not None:
        payload["description"] = description
    meta_path_for(directory, filename).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def read_meta(directory: Path, filename: str):
    path = meta_path_for(directory, filename)
    if not path.is_file():
        return {"source": "upload", "description": None}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"source": "upload", "description": None}

    source = payload.get("source") if isinstance(payload, dict) else "upload"
    description = payload.get("description") if isinstance(payload, dict) else None
    if source not in {"upload", "design"}:
        source = "upload"
    if description is not None and not isinstance(description, str):
        description = str(description)
    return {"source": source, "description": description}




def user_audio_dir(user: str) -> Path:
    candidate = (user or "default").strip()
    if not candidate or len(candidate) > 32 or not re.fullmatch(r"[A-Za-z0-9_-]+", candidate):
        raise HTTPException(status_code=400, detail="Invalid user. Use up to 32 chars: letters, numbers, underscore, hyphen")
    directory = REFERENCE_AUDIO_DIR / candidate
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def user_presets_path(user: str) -> Path:
    return user_audio_dir(user) / "voice_presets.json"


voice_presets_lock = threading.Lock()


def _normalize_preset_name(name: str) -> str:
    normalized = (name or "").strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="Preset name is required")
    return normalized


def _normalize_voice_preset_payload(payload: dict) -> dict:
    name = _normalize_preset_name(str(payload.get("name", "")))
    voice_description = str(payload.get("voice_description", "")).strip()
    if not voice_description:
        raise HTTPException(status_code=400, detail="voice_description is required")
    description = payload.get("description")
    if description is not None:
        description = str(description).strip() or None
    return {
        "name": name,
        "voice_description": voice_description,
        "description": description,
    }


def load_voice_presets(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    presets: dict[str, dict] = {}
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                try:
                    normalized = _normalize_voice_preset_payload(item)
                    presets[normalized["name"]] = normalized
                except HTTPException:
                    continue
    elif isinstance(raw, dict):
        for key, value in raw.items():
            if isinstance(value, dict):
                candidate = {"name": key, **value}
            else:
                candidate = {"name": key, "voice_description": value}
            try:
                normalized = _normalize_voice_preset_payload(candidate)
                presets[normalized["name"]] = normalized
            except HTTPException:
                continue

    return presets


def save_voice_presets(path: Path, presets: dict[str, dict]):
    ordered = [presets[name] for name in sorted(presets)]
    path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")


def apply_voice_preset(req: TTSRequest) -> TTSRequest:
    if not req.voice_preset:
        return req

    preset_name = _normalize_preset_name(req.voice_preset)
    with voice_presets_lock:
        presets = load_voice_presets(user_presets_path(req.user or "default"))
    preset = presets.get(preset_name)
    if not preset:
        raise HTTPException(status_code=404, detail=f"voice preset not found: {preset_name}")

    return req.model_copy(
        update={
            "mode": "design",
            "voice_description": preset["voice_description"],
            "voice_preset": preset_name,
        }
    )


IDLE_CUDA_RELEASE_SECONDS = float(os.getenv("QWEN_IDLE_CUDA_RELEASE_SECONDS", "15"))
IDLE_CUDA_RELEASE_MIN_MB = int(os.getenv("QWEN_IDLE_CUDA_RELEASE_MIN_MB", "64"))


def _idle_cuda_release_loop() -> None:
    """Hand cached CUDA memory back to the GPU while no model is loaded.

    unload_model() empties the cache, but a finished request can still hold
    tensors at that moment (e.g. the audio-stream decoder keeps the speech
    tokenizer alive until the response object is collected). Those blocks
    land back in PyTorch's cache afterwards and would otherwise stay
    reserved, starving other GPU tenants such as llama-server.
    """
    while True:
        time.sleep(IDLE_CUDA_RELEASE_SECONDS)
        try:
            svc.release_idle_cuda_memory(IDLE_CUDA_RELEASE_MIN_MB)
        except Exception as e:
            print(json.dumps({"event": "tts_idle_cuda_release_error", "detail": str(e)}), flush=True)


@app.on_event("startup")
def start_idle_cuda_release():
    if IDLE_CUDA_RELEASE_SECONDS > 0 and torch.cuda.is_available():
        threading.Thread(target=_idle_cuda_release_loop, name="idle-cuda-release", daemon=True).start()


@app.on_event("startup")
def ensure_reference_audio_dir():
    REFERENCE_AUDIO_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/healthz")
def healthz():
    return {"ok": True, "uptime_s": int(time.time() - start_time)}


def normalize_asr_language(language: Optional[str]) -> Optional[str]:
    raw = (language or "").strip()
    if not raw or raw.lower() == "auto":
        return None
    lowered = raw.lower()
    if lowered in _ASR_LANGUAGE_BY_LOWER:
        return _ASR_LANGUAGE_BY_LOWER[lowered]
    if lowered in ASR_LANGUAGE_ALIASES:
        return ASR_LANGUAGE_ALIASES[lowered]
    primary = re.split(r"[-_]", lowered, maxsplit=1)[0]
    if primary in ASR_LANGUAGE_ALIASES:
        return ASR_LANGUAGE_ALIASES[primary]
    raise HTTPException(status_code=400, detail=f"Unsupported ASR language: {raw}")


def _resolve_openai_asr_model_size(model: Optional[str]) -> str:
    raw = (model or "").strip().lower()
    if not raw:
        return "fast"
    if raw in ASR_MODEL_IDS:
        return raw
    if raw == ASR_MODEL_IDS["quality"].lower() or "1.7b" in raw:
        return "quality"
    if raw == ASR_MODEL_IDS["fast"].lower() or "0.6b" in raw:
        return "fast"
    return "fast"


def read_uploaded_asr_bytes(file: UploadFile) -> bytes:
    max_bytes = ASR_MAX_UPLOAD_MB * 1024 * 1024
    data = file.file.read(max_bytes + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty")
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Uploaded audio exceeds {ASR_MAX_UPLOAD_MB} MB limit")
    return data


def _decode_asr_audio_ffmpeg(data: bytes, filename: str = "") -> np.ndarray:
    suffix = Path(filename).suffix.lower() if filename else ""
    if not re.fullmatch(r"\.[a-z0-9]{1,8}", suffix):
        suffix = ".audio"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        try:
            proc = subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel", "error",
                    "-i", tmp.name,
                    "-vn",
                    "-ac", "1",
                    "-ar", str(ASR_SAMPLE_RATE),
                    "-f", "f32le",
                    "pipe:1",
                ],
                capture_output=True,
                check=False,
            )
        except FileNotFoundError:
            raise HTTPException(
                status_code=500,
                detail="ffmpeg is required to decode this audio format",
            )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise HTTPException(
            status_code=400,
            detail=f"Failed to decode audio{f': {err}' if err else ''}",
        )
    return np.frombuffer(proc.stdout, dtype="<f4").copy()


def decode_asr_audio(data: bytes, filename: str = "") -> tuple[np.ndarray, int]:
    """Decode uploaded audio bytes to 16 kHz mono float32 PCM."""
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty")

    wav: Optional[np.ndarray] = None
    sf_decoded = None
    try:
        raw, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        mono = raw.mean(axis=1) if raw.shape[1] > 1 else raw[:, 0]
        if int(sr) == ASR_SAMPLE_RATE:
            wav = np.asarray(mono, dtype=np.float32)
        else:
            sf_decoded = (np.asarray(mono, dtype=np.float32), int(sr))
    except Exception:
        sf_decoded = None

    if wav is None:
        try:
            wav = _decode_asr_audio_ffmpeg(data, filename)
        except HTTPException as exc:
            if exc.status_code == 500 and sf_decoded is not None:
                import librosa

                mono, sr = sf_decoded
                wav = np.asarray(
                    librosa.resample(y=mono, orig_sr=sr, target_sr=ASR_SAMPLE_RATE),
                    dtype=np.float32,
                )
            else:
                raise

    wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    if wav.size == 0:
        raise HTTPException(status_code=400, detail="Decoded audio is empty")
    return wav, ASR_SAMPLE_RATE


@app.get("/status")
def status():
    return {
        "model_state": model_status["state"],
        "model_kind": model_status["kind"],
        "model_id": model_status["model_id"],
        "device": model_status["device"],
        "uptime_s": int(time.time() - start_time),
        "cuda": cuda_memory_snapshot(),
    }


@app.get("/readyz")
def readyz():
    if model_status["state"] == "ready":
        return {
            "ready": True,
            "model_kind": model_status["kind"],
            "model_id": model_status["model_id"],
            "device": model_status["device"],
        }
    return JSONResponse(
        status_code=503,
        content={"ready": False, "reason": model_status["state"]},
    )


@app.get("/info")
def info():
    return {
        "models": MODEL_IDS,
        "modes": ["clone", "design"],
        "sizes": ["fast", "quality"],
        "current_model": svc._model_id,
        "current_model_kind": svc._model_kind,
        "current_device": svc._model_device,
        "supported_languages": SUPPORTED_LANGUAGES,
        "asr": {
            "available": Qwen3ASRModel is not None,
            "models": ASR_MODEL_IDS,
            "sizes": ["fast", "quality"],
            "supported_languages": ASR_SUPPORTED_LANGUAGES,
            "max_new_tokens": svc.asr_max_new_tokens,
            "max_inference_batch_size": svc.asr_max_inference_batch_size,
            "max_upload_mb": ASR_MAX_UPLOAD_MB,
        },
        "generation": {
            "max_chunk_chars": MAX_CHUNK_CHARS,
            "stream_first_frames": STREAM_FIRST_FRAMES,
            "stream_step_frames": STREAM_STEP_FRAMES,
            "trim_chunk_silence": TRIM_CHUNK_SILENCE,
            "trim_silence_keep_ms": TRIM_SILENCE_KEEP_MS,
            "chunk_fade_ms": CHUNK_FADE_MS,
            "max_new_tokens_hard_cap": svc.max_new_tokens,
            "max_new_tokens_per_char": svc.max_new_tokens_per_char,
            "max_new_tokens_buffer": svc.max_new_tokens_buffer,
            "non_streaming_mode": svc.non_streaming_mode,
            "idle_unload_seconds": svc.idle_unload_seconds,
            "speaker_embedding_cache_max_items": svc.speaker_cache_max_items,
            "default_device": svc.device,
            "cpu_mode_supported": True,
            "cpu_dtype": svc.cpu_dtype,
            "cpu_attn_impl": svc.cpu_attn_impl,
            "speech_tokenizer_device": svc.speech_tokenizer_device,
            "speaker_encoder_device": svc.speaker_encoder_device,
            "qwen_memory_patch": QWEN_MEMORY_PATCH_ENABLED,
        },
    }


@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    svc.hard_reset()

    response = {"ok": True, "state": "idle", "model_id": None, "device": None}
    if req and req.restart:
        exit_code = int(os.getenv("QWEN_RESET_EXIT_CODE", "1"))

        def _delayed_exit():
            time.sleep(0.2)
            os._exit(exit_code)

        threading.Thread(target=_delayed_exit, daemon=True).start()
        response["restart_scheduled"] = True
        response["exit_code"] = exit_code

    return response


@app.get("/reference-audio")
def list_reference_audio(user: str = Query(default="default")):
    dir_path = user_audio_dir(user)
    files = []
    for p in sorted(dir_path.iterdir()):
        if not p.is_file() or p.name.startswith("."):
            continue
        if p.name.endswith(".meta.json"):
            continue
        meta = read_meta(dir_path, p.name)
        files.append(
            {
                "filename": p.name,
                "source": meta.get("source", "upload"),
                "description": meta.get("description"),
            }
        )

    return {"files": files}


@app.post("/reference-audio/upload")
async def upload_reference_audio(file: UploadFile = File(...), user: str = Query(default="default")):
    dir_path = user_audio_dir(user)
    filename = Path(file.filename or "").name
    if not filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    target = (dir_path / filename).resolve()
    if not str(target).startswith(str(dir_path.resolve())):
        raise HTTPException(status_code=400, detail="Invalid filename")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    target.write_bytes(data)
    write_meta(dir_path, filename, source="upload")
    return {"ok": True, "filename": filename, "size": len(data)}


@app.post("/reference-audio/save-design")
def save_design_reference_audio(req: SaveDesignRequest, user: str = Query(default="default")):
    dir_path = user_audio_dir(user)
    filename = Path(req.filename or "").name
    if not filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    target = (dir_path / filename).resolve()
    if not str(target).startswith(str(dir_path.resolve())):
        raise HTTPException(status_code=400, detail="Invalid filename")

    audio_b64 = req.audio_b64.strip()
    if "," in audio_b64 and audio_b64.lower().startswith("data:"):
        audio_b64 = audio_b64.split(",", 1)[1]

    try:
        data = base64.b64decode(audio_b64, validate=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio_b64: {e}")

    if not data:
        raise HTTPException(status_code=400, detail="Decoded audio is empty")

    target.write_bytes(data)
    write_meta(dir_path, filename, source="design", description=(req.description or "").strip() or None)
    return {"ok": True, "filename": filename}


@app.delete("/reference-audio/{filename}")
def delete_reference_audio(filename: str, user: str = Query(default="default")):
    dir_path = user_audio_dir(user)
    target = svc.resolve_reference_audio(filename, dir_path)
    target.unlink()
    sidecar = meta_path_for(dir_path, target.name)
    if sidecar.exists():
        sidecar.unlink()
    return {"ok": True, "deleted": target.name}


@app.get("/voice-presets")
def list_voice_presets(user: str = Query(default="default")):
    with voice_presets_lock:
        presets = load_voice_presets(user_presets_path(user))
        items = [presets[name] for name in sorted(presets)]
    return {"presets": items}


@app.post("/voice-presets")
def upsert_voice_preset(req: VoicePresetRequest, user: str = Query(default="default")):
    normalized = _normalize_voice_preset_payload(req.model_dump())
    with voice_presets_lock:
        path = user_presets_path(user)
        presets = load_voice_presets(user_presets_path(user))
        presets[normalized["name"]] = normalized
        save_voice_presets(path, presets)
    return {"ok": True, "preset": normalized}


@app.delete("/voice-presets/{name}")
def delete_voice_preset(name: str, user: str = Query(default="default")):
    preset_name = _normalize_preset_name(name)
    with voice_presets_lock:
        path = user_presets_path(user)
        presets = load_voice_presets(user_presets_path(user))
        if preset_name not in presets:
            raise HTTPException(status_code=404, detail=f"voice preset not found: {preset_name}")
        del presets[preset_name]
        save_voice_presets(path, presets)

    return {"ok": True, "deleted": preset_name}


def _build_atempo_filter(speed: float) -> Optional[str]:
    if abs(speed - 1.0) <= 1e-6:
        return None
    factors: list[float] = []
    remaining = float(speed)
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    return ",".join(f"atempo={f:.6f}" for f in factors)


def _ffmpeg_encode_pcm_stream(
    pcm_chunks,
    sample_rate: int,
    target_format: str,
    speed: float,
    cancel: threading.Event,
):
    """Stream raw s16le mono PCM through ffmpeg into the requested format."""
    atempo = _build_atempo_filter(speed)
    if target_format == "mp3":
        out_args = ["-f", "mp3", "-c:a", "libmp3lame", "-b:a", "128k", "-write_xing", "0", "-id3v2_version", "0"]
    elif target_format == "opus":
        out_args = ["-f", "ogg", "-c:a", "libopus", "-b:a", "64k", "-page_duration", "40000"]
    elif target_format == "aac":
        out_args = ["-f", "adts", "-c:a", "aac", "-b:a", "128k"]
    elif target_format == "flac":
        out_args = ["-f", "flac", "-c:a", "flac"]
    elif target_format in {"pcm", "wav"}:
        out_args = ["-f", "s16le", "-c:a", "pcm_s16le"]
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported audio format: {target_format}")

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-f", "s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        "-i", "pipe:0",
    ]
    if atempo:
        cmd.extend(["-af", atempo])
    cmd.extend(["-flush_packets", "1", *out_args, "pipe:1"])

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail=f"ffmpeg is required to stream {target_format} audio",
        )

    feed_error: list[Optional[BaseException]] = [None]
    stderr_chunks: list[bytes] = []

    def feed_stdin() -> None:
        try:
            for chunk in pcm_chunks:
                if cancel.is_set():
                    break
                if not chunk:
                    continue
                try:
                    proc.stdin.write(chunk)
                    proc.stdin.flush()
                except (BrokenPipeError, OSError):
                    break
        except BaseException as exc:
            feed_error[0] = exc
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass

    def drain_stderr() -> None:
        try:
            data = proc.stderr.read()
            if data:
                stderr_chunks.append(data)
        except Exception:
            pass
        finally:
            try:
                proc.stderr.close()
            except Exception:
                pass

    feeder = threading.Thread(target=feed_stdin, daemon=True)
    err_reader = threading.Thread(target=drain_stderr, daemon=True)
    feeder.start()
    err_reader.start()

    stdout_fd = proc.stdout.fileno()
    completed = False
    try:
        while True:
            data = os.read(stdout_fd, 4096)
            if not data:
                break
            yield data
        completed = True
    finally:
        if not completed:
            cancel.set()
            if proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass
        try:
            proc.stdout.close()
        except Exception:
            pass
        if proc.poll() is None:
            try:
                proc.wait(timeout=5 if completed else 2)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except Exception:
                    pass
                proc.wait()
        feeder.join(timeout=5)
        err_reader.join(timeout=2)

    if feed_error[0] is not None:
        raise feed_error[0]
    if not cancel.is_set() and proc.returncode != 0:
        err_msg = b"".join(stderr_chunks).decode("utf-8", errors="replace").strip()
        raise HTTPException(
            status_code=500,
            detail=f"ffmpeg failed while encoding {target_format}{f': {err_msg}' if err_msg else ''}",
        )


def _resolve_openai_tts_model_size(model: Optional[str]) -> str:
    raw = (model or "").strip().lower()
    if not raw:
        return "fast"
    if raw in {"fast", "quality"}:
        return raw
    if (
        raw in {"tts-1-hd", "hd"}
        or raw == MODEL_IDS["clone"]["quality"].lower()
        or raw == MODEL_IDS["design"]["quality"].lower()
        or "1.7b" in raw
        or "hd" in raw.split("-")
    ):
        return "quality"
    return "fast"


def _find_voice_preset_by_voice(voice: str, user: str) -> Optional[dict]:
    with voice_presets_lock:
        presets = load_voice_presets(user_presets_path(user))
    if voice in presets:
        return presets[voice]
    lowered = voice.lower()
    for name, preset in presets.items():
        if name.lower() == lowered:
            return preset
    return None


def _find_reference_audio_by_voice(voice: str, user_dir: Path) -> Optional[str]:
    if not voice or "/" in voice or "\\" in voice or voice.startswith(".") or ".." in voice:
        return None
    exact = user_dir / voice
    if exact.is_file() and not exact.name.endswith(".meta.json"):
        return exact.name
    lowered = voice.lower()
    ci_match: Optional[str] = None
    for p in sorted(user_dir.iterdir()):
        if not p.is_file() or p.name.startswith(".") or p.name.endswith(".meta.json"):
            continue
        if p.stem == voice:
            return p.name
        if ci_match is None and (p.name.lower() == lowered or p.stem.lower() == lowered):
            ci_match = p.name
    return ci_match


def _combine_voice_description(base: str, instructions: str) -> str:
    base = base.strip()
    instructions = instructions.strip()
    if not instructions:
        return base
    if not base:
        return instructions
    sep = " " if base[-1] in ".!?" else ". "
    return f"{base}{sep}{instructions}"


def _resolve_openai_voice(
    voice: Optional[str],
    instructions: Optional[str],
    user: str,
    user_dir: Path,
    _allow_default: bool = True,
) -> dict:
    raw_voice = (voice or "").strip()
    instr = (instructions or "").strip()
    lowered = raw_voice.lower()

    if lowered.startswith("preset:"):
        preset_name = _normalize_preset_name(raw_voice[len("preset:"):])
        preset = _find_voice_preset_by_voice(preset_name, user)
        if not preset:
            raise HTTPException(status_code=404, detail=f"voice preset not found: {preset_name}")
        return {
            "mode": "design",
            "voice_preset": preset["name"],
            "voice_description": _combine_voice_description(preset["voice_description"], instr),
            "reference_audio": None,
        }

    if lowered.startswith("clone:"):
        ref_name = raw_voice[len("clone:"):].strip()
        if not ref_name:
            raise HTTPException(status_code=400, detail="reference audio filename required after clone:")
        resolved = _find_reference_audio_by_voice(ref_name, user_dir)
        if not resolved:
            svc.resolve_reference_audio(ref_name, user_dir)
            resolved = ref_name
        return {
            "mode": "clone",
            "voice_preset": None,
            "voice_description": None,
            "reference_audio": resolved,
        }

    if lowered.startswith("design:"):
        desc = _combine_voice_description(raw_voice[len("design:"):], instr)
        if not desc:
            raise HTTPException(status_code=400, detail="voice description required after design:")
        return {
            "mode": "design",
            "voice_preset": None,
            "voice_description": desc,
            "reference_audio": None,
        }

    if raw_voice:
        preset = _find_voice_preset_by_voice(raw_voice, user)
        if preset:
            return {
                "mode": "design",
                "voice_preset": preset["name"],
                "voice_description": _combine_voice_description(preset["voice_description"], instr),
                "reference_audio": None,
            }

        ref_file = _find_reference_audio_by_voice(raw_voice, user_dir)
        if ref_file:
            return {
                "mode": "clone",
                "voice_preset": None,
                "voice_description": None,
                "reference_audio": ref_file,
            }

    if not raw_voice or lowered == "default" or lowered in OPENAI_VOICE_DESCRIPTIONS:
        default_voice = os.getenv("QWEN_OPENAI_DEFAULT_VOICE", "").strip()
        if _allow_default and default_voice:
            return _resolve_openai_voice(default_voice, instr, user, user_dir, _allow_default=False)
        if not raw_voice or lowered == "default":
            base_desc = instr if instr else OPENAI_VOICE_DESCRIPTIONS["alloy"]
            return {
                "mode": "design",
                "voice_preset": None,
                "voice_description": base_desc,
                "reference_audio": None,
            }
        return {
            "mode": "design",
            "voice_preset": None,
            "voice_description": _combine_voice_description(OPENAI_VOICE_DESCRIPTIONS[lowered], instr),
            "reference_audio": None,
        }

    if " " in raw_voice:
        return {
            "mode": "design",
            "voice_preset": None,
            "voice_description": _combine_voice_description(raw_voice, instr),
            "reference_audio": None,
        }

    raise HTTPException(
        status_code=404,
        detail=(
            f"Voice not found: {raw_voice}. Use a voice preset name, reference audio filename, "
            f"standard OpenAI voice ({', '.join(sorted(OPENAI_VOICE_DESCRIPTIONS))}), or a voice description."
        ),
    )


def _stream_tts_audio(
    req: TTSRequest,
    user_dir: Path,
    audio_format: str,
    speed: float = 1.0,
    stream_format: str = "audio",
    log_prefix: str = "tts_audio_stream",
    content_disposition: Optional[str] = None,
) -> StreamingResponse:
    """Shared frame-level streaming TTS engine for /api/tts/audio-stream and /v1/audio/speech."""
    if not QWEN_MEMORY_PATCH_ENABLED:
        raise HTTPException(status_code=501, detail="frame streaming unavailable: talker patch not active")
    svc.validate_request(req, user_dir)

    chunks = chunk_text(req.text)
    n_chunks = len(chunks)
    t0 = time.time()
    session = svc.begin_request()
    try:
        base_extra_kwargs = svc.prepare_request_extra_kwargs(req, user_dir)
        model = svc.loaded_model()
        speech_tokenizer = model.model.speech_tokenizer
        if speech_tokenizer.get_model_type() != "qwen3_tts_tokenizer_12hz":
            raise HTTPException(status_code=501, detail="frame streaming requires a 12Hz model")
        sample_rate = int(speech_tokenizer.get_output_sample_rate())
    except Exception:
        session.finish(unload_now=True)
        raise

    events: queue.Queue = queue.Queue()
    cancel = threading.Event()

    def on_frame(frame: torch.Tensor) -> None:
        if cancel.is_set():
            raise StreamCancelled()
        events.put(("frame", frame))

    def run_generation():
        try:
            for i, chunk in enumerate(chunks):
                if cancel.is_set():
                    break
                events.put(("chunk_start", i))
                svc.synthesize(
                    req.model_copy(update={"text": chunk}),
                    user_dir,
                    extra_kwargs={**base_extra_kwargs, "non_streaming_mode": False},
                    on_frame=on_frame,
                )
                events.put(("chunk_end", i))
            events.put(("done", None))
        except StreamCancelled:
            events.put(("done", None))
        except Exception as e:
            events.put(("error", e))

    stats = {"frames": 0, "samples": 0, "t_first_audio_ms": None}

    def audio_pieces():
        """Yield float audio arrays in playback order."""
        decoder: Optional[StreamingCodecDecoder] = None
        shaper: Optional[ChunkEdgeShaper] = None
        chunk_frames = 0
        t_chunk = time.time()
        while True:
            batch = [events.get()]
            while True:
                try:
                    batch.append(events.get_nowait())
                except queue.Empty:
                    break
            for kind, value in batch:
                if kind == "chunk_start":
                    decoder = StreamingCodecDecoder(speech_tokenizer)
                    shaper = ChunkEdgeShaper(sample_rate)
                    chunk_frames = 0
                    t_chunk = time.time()
                elif kind == "frame":
                    decoder.push(value)
                    chunk_frames += 1
                elif kind == "chunk_end":
                    yield shaper.feed(decoder.decode_pending())
                    yield shaper.finish()
                    print(json.dumps({
                        "event": f"{log_prefix}_chunk",
                        "chunk": value + 1,
                        "of": n_chunks,
                        "chars": len(chunks[value]),
                        "frames": chunk_frames,
                        "audio_ms": chunk_frames * decoder.upsample * 1000 // sample_rate,
                        "t_chunk_ms": int((time.time() - t_chunk) * 1000),
                        "cuda": cuda_memory_snapshot(),
                    }), flush=True)
                    stats["frames"] += chunk_frames
                    decoder = shaper = None
                elif kind == "error":
                    raise value
                elif kind == "done":
                    return
            if decoder is not None and decoder.ready():
                yield shaper.feed(decoder.decode_pending())

    def pcm_payloads():
        for piece in audio_pieces():
            if piece.size:
                if stats["t_first_audio_ms"] is None:
                    stats["t_first_audio_ms"] = int((time.time() - t0) * 1000)
                stats["samples"] += int(piece.size)
                yield float_wav_to_pcm16_bytes(piece)

    gen_thread = threading.Thread(target=run_generation, daemon=True)
    gen_thread.start()
    payloads = pcm_payloads()

    # Wait for the first audio before sending headers, so a failure (GPU
    # busy, bad voice, ...) is a proper error status, not an empty 200.
    try:
        first_pcm = next(payloads, b"")
    except Exception as e:
        cancel.set()
        gen_thread.join()
        payloads.close()
        print(json.dumps({
            "event": f"{log_prefix}_first_audio_error",
            "detail": str(getattr(e, "detail", e)),
            "chunks": n_chunks,
            "cuda": cuda_memory_snapshot(),
        }), flush=True)
        session.finish(unload_now=True)
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"tts audio stream failed before first audio: {e}")

    needs_ffmpeg = audio_format not in {"pcm", "wav"} or abs(speed - 1.0) > 1e-6
    encoded_stream = None
    first_out = first_pcm

    if needs_ffmpeg and first_pcm:
        encoded_stream = _ffmpeg_encode_pcm_stream(
            itertools.chain([first_pcm], payloads),
            sample_rate=sample_rate,
            target_format=audio_format,
            speed=speed,
            cancel=cancel,
        )
        try:
            first_out = next(encoded_stream, b"")
        except Exception as e:
            cancel.set()
            encoded_stream.close()
            gen_thread.join()
            payloads.close()
            print(json.dumps({
                "event": f"{log_prefix}_first_audio_error",
                "detail": str(getattr(e, "detail", e)),
                "chunks": n_chunks,
                "cuda": cuda_memory_snapshot(),
            }), flush=True)
            session.finish(unload_now=True)
            if isinstance(e, HTTPException):
                raise
            raise HTTPException(status_code=500, detail=f"tts audio stream failed before first audio: {e}")

    stream_error = [False]

    def audio_bytes_body():
        if audio_format == "wav":
            yield wav_stream_header(sample_rate)
        if first_out:
            yield first_out
        if encoded_stream is not None:
            yield from encoded_stream
        elif first_pcm:
            yield from payloads

    def body():
        try:
            if stream_format == "sse":
                for chunk_bytes in audio_bytes_body():
                    if not chunk_bytes:
                        continue
                    delta = {
                        "type": "speech.audio.delta",
                        "audio": base64.b64encode(chunk_bytes).decode("ascii"),
                    }
                    yield f"event: speech.audio.delta\ndata: {json.dumps(delta)}\n\n".encode("utf-8")
                input_tokens = max(1, len((req.text or "").split()))
                output_tokens = int(stats["frames"])
                done = {
                    "type": "speech.audio.done",
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                    },
                }
                yield f"event: speech.audio.done\ndata: {json.dumps(done)}\n\n".encode("utf-8")
            else:
                yield from audio_bytes_body()
            print(json.dumps({
                "event": f"{log_prefix}_done",
                "mode": req.mode,
                "format": audio_format,
                "stream_format": stream_format,
                "speed": speed,
                "chars": len(req.text or ""),
                "chunks": n_chunks,
                "frames": stats["frames"],
                "audio_ms": stats["samples"] * 1000 // sample_rate,
                "t_first_audio_ms": stats["t_first_audio_ms"],
                "t_total_ms": int((time.time() - t0) * 1000),
                "cuda": cuda_memory_snapshot(),
            }), flush=True)
        except Exception as e:
            stream_error[0] = True
            print(json.dumps({
                "event": f"{log_prefix}_error",
                "detail": str(getattr(e, "detail", e)),
                "chunks": n_chunks,
                "cuda": cuda_memory_snapshot(),
            }), flush=True)
            raise

    def close_stream() -> None:
        # Stop generation at the next frame, stop ffmpeg if active, wait for
        # the worker to exit generate(), drop decoder tensors, then flush CUDA
        # cache and hand off to the idle-unload timer (or unload on error).
        cancel.set()
        if encoded_stream is not None:
            encoded_stream.close()
        gen_thread.join()
        payloads.close()
        session.finish(unload_now=stream_error[0])

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "X-Sample-Rate": str(sample_rate),
        "X-Channels": "1",
    }
    if audio_format in {"pcm", "wav"}:
        headers["X-Sample-Format"] = "s16le"
    if content_disposition:
        headers["Content-Disposition"] = content_disposition

    media_type = "text/event-stream" if stream_format == "sse" else OPENAI_AUDIO_CONTENT_TYPES[audio_format]
    return StreamingResponse(
        _ManagedStream(body(), close_stream),
        media_type=media_type,
        headers=headers,
    )


@app.post("/api/tts/audio-stream")
def tts_audio_stream(req: TTSRequest):
    """Stream audio while it is being generated.

    Codec frames are captured from the talker as they are sampled and decoded
    in small spans, so playback can start after a few hundred ms of audio
    instead of after a whole text chunk. Long text is still split with
    chunk_text(); each chunk is generated in turn on the same stream.

    `audio_format`: `pcm` = raw s16le mono, `wav` = the same with a streaming
    WAV header. The sample rate is in the X-Sample-Rate header either way.
    """
    req = apply_voice_preset(req)
    user_dir = user_audio_dir(req.user or "default")
    audio_format = req.audio_format.lower().strip()
    if audio_format not in {"wav", "pcm"}:
        raise HTTPException(status_code=400, detail="audio_format must be wav or pcm for audio-stream endpoint")
    return _stream_tts_audio(
        req=req,
        user_dir=user_dir,
        audio_format=audio_format,
        speed=1.0,
        stream_format="audio",
        log_prefix="tts_audio_stream",
        content_disposition='inline; filename="tts-stream.wav"' if audio_format == "wav" else None,
    )


@app.post("/v1/audio/speech")
def openai_audio_speech(req: OpenAISpeechRequest):
    """OpenAI-compatible streaming speech synthesis endpoint backed by Qwen3-TTS."""
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="input must not be empty")

    audio_format = (req.response_format or "mp3").strip().lower()
    if audio_format not in OPENAI_AUDIO_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"response_format must be one of: {', '.join(OPENAI_AUDIO_CONTENT_TYPES)}",
        )

    stream_format = (req.stream_format or "audio").strip().lower()
    if stream_format not in {"audio", "sse"}:
        raise HTTPException(status_code=400, detail="stream_format must be audio or sse")

    user = req.user or "default"
    user_dir = user_audio_dir(user)
    resolved_voice = _resolve_openai_voice(
        voice=req.voice,
        instructions=req.instructions,
        user=user,
        user_dir=user_dir,
    )
    tts_req = TTSRequest(
        text=req.input,
        language=req.language or "Auto",
        audio_format="pcm",
        mode=resolved_voice["mode"],
        model_size=_resolve_openai_tts_model_size(req.model),
        reference_audio=resolved_voice["reference_audio"],
        voice_description=resolved_voice["voice_description"],
        voice_preset=resolved_voice["voice_preset"],
        user=user,
        cpu_mode=req.cpu_mode,
    )
    return _stream_tts_audio(
        req=tts_req,
        user_dir=user_dir,
        audio_format=audio_format,
        speed=req.speed,
        stream_format=stream_format,
        log_prefix="openai_speech",
    )


@app.post("/api/tts/stream")
def tts_stream(req: TTSRequest):
    req = apply_voice_preset(req)
    user_dir = user_audio_dir(req.user or "default")
    audio_format = req.audio_format.lower().strip()
    if audio_format != "wav":
        raise HTTPException(status_code=400, detail="audio_format must be wav for stream endpoint")
    svc.validate_request(req, user_dir)

    chunks = chunk_text(req.text)
    n_chunks = len(chunks)
    session = svc.begin_request()
    try:
        base_extra_kwargs = svc.prepare_request_extra_kwargs(req, user_dir)
    except Exception:
        session.finish(unload_now=True)
        raise

    q: queue.Queue = queue.Queue()

    def run_generation():
        unload_now = False
        try:
            wav_arrays: list[np.ndarray] = []
            sr_final = 24000
            total_tokens = 0

            for chunk_idx, chunk in enumerate(chunks):
                chunk_estimated = max(1, svc._max_new_tokens_for_text(chunk))
                chunk_token_count = [0]

                q.put(("chunk_start", {
                    "chunk": chunk_idx + 1,
                    "of": n_chunks,
                    "chars": len(chunk),
                }))

                class ProgressProcessor(LogitsProcessor):
                    def __call__(self, input_ids, scores):
                        chunk_token_count[0] += 1
                        # Progress within this chunk (0–95%), scaled to its share of total
                        chunk_pct = min(95, int((chunk_token_count[0] / chunk_estimated) * 100))
                        # Overall progress across all chunks
                        overall_pct = int(((chunk_idx + chunk_pct / 100) / n_chunks) * 100)
                        q.put(("progress", {
                            "chunk": chunk_idx + 1,
                            "of": n_chunks,
                            "chunk_pct": chunk_pct,
                            "overall_pct": min(95, overall_pct),
                            "tokens": chunk_token_count[0],
                        }))
                        return scores

                chunk_req = req.model_copy(update={"text": chunk})
                wav, sr, _used_model_id, _used_size = svc.synthesize(
                    chunk_req,
                    user_dir,
                    extra_kwargs={
                        **base_extra_kwargs,
                        "logits_processor": LogitsProcessorList([ProgressProcessor()]),
                    },
                )
                wav = postprocess_chunk_audio(wav, int(sr))
                wav_arrays.append(np.asarray(wav, dtype=np.float32))
                del wav
                sr_final = sr
                total_tokens += chunk_token_count[0]

                q.put(("chunk_done", {
                    "chunk": chunk_idx + 1,
                    "of": n_chunks,
                    "tokens": chunk_token_count[0],
                }))

            data = np.concatenate(wav_arrays) if len(wav_arrays) > 1 else wav_arrays[0]
            buf = io.BytesIO()
            sf.write(buf, data, sr_final, format="WAV")
            audio_b64 = base64.b64encode(buf.getvalue()).decode()
            q.put(("done", {
                "audio_b64": audio_b64,
                "chunks": n_chunks,
                "tokens_generated": total_tokens,
            }))
        except Exception as e:
            unload_now = True
            detail = e.detail if isinstance(e, HTTPException) else str(e)
            payload = {"detail": detail}
            if isinstance(e, HTTPException):
                payload["status_code"] = e.status_code
            q.put(("error", payload))
        finally:
            session.finish(unload_now=unload_now)

    threading.Thread(target=run_generation, daemon=True).start()

    def event_stream():
        while True:
            event_type, payload = q.get()
            yield f"event: {event_type}\ndata: {json.dumps(payload)}\n\n".encode()
            if event_type in {"done", "error"}:
                break

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/tts")
def tts(req: TTSRequest):
    req = apply_voice_preset(req)
    user_dir = user_audio_dir(req.user or "default")
    t0 = time.time()
    audio_format = req.audio_format.lower().strip()
    if audio_format not in {"wav", "ogg"}:
        raise HTTPException(status_code=400, detail="audio_format must be wav or ogg")
    svc.validate_request(req, user_dir)

    chunks = chunk_text(req.text)
    n_chunks = len(chunks)
    wav_arrays: list[np.ndarray] = []
    sr_final: int = 24000
    used_model_id: str = ""
    used_size: str = ""

    with svc.request_session():
        base_extra_kwargs = svc.prepare_request_extra_kwargs(req, user_dir)

        t1 = time.time()
        for i, chunk in enumerate(chunks):
            chunk_req = req.model_copy(update={"text": chunk})
            wav, sr, used_model_id, used_size = svc.synthesize(
                chunk_req, user_dir, extra_kwargs=base_extra_kwargs
            )
            wav = postprocess_chunk_audio(wav, int(sr))
            wav_arrays.append(np.asarray(wav, dtype=np.float32))
            del wav
            sr_final = sr
            if n_chunks > 1:
                print(json.dumps({"event": "tts_chunk", "chunk": i + 1, "of": n_chunks, "chars": len(chunk)}))

    data = np.concatenate(wav_arrays) if len(wav_arrays) > 1 else wav_arrays[0]
    t2 = time.time()

    buf = io.BytesIO()
    if audio_format == "wav":
        sf.write(buf, data, sr_final, format="WAV")
        mime = "audio/wav"
    else:
        sf.write(buf, data, sr_final, format="OGG", subtype="VORBIS")
        mime = "audio/ogg"
    t3 = time.time()

    total = t3 - t0
    t_synth = t2 - t1
    t_encode = t3 - t2

    try:
        print(
            json.dumps(
                {
                    "event": "tts_timing",
                    "mode": req.mode,
                    "model_size": used_size,
                    "chars": len(req.text or ""),
                    "chunks": n_chunks,
                    "model_id": used_model_id,
                    "t_total_ms": int(total * 1000),
                    "t_synth_ms": int(t_synth * 1000),
                    "t_encode_ms": int(t_encode * 1000),
                }
            )
        )
    except Exception:
        pass

    return Response(content=buf.getvalue(), media_type=mime)


def _run_asr_request(
    file: UploadFile,
    model_size: Optional[str],
    language: Optional[str],
    context: str,
    cpu_mode: bool,
) -> dict:
    if Qwen3ASRModel is None:
        raise HTTPException(status_code=501, detail="ASR unavailable: qwen-asr package is not installed")
    t0 = time.time()
    normalized_language = normalize_asr_language(language)
    data = read_uploaded_asr_bytes(file)
    wav, sr = decode_asr_audio(data, file.filename or "")
    duration_s = round(float(wav.size) / float(sr), 3)

    with svc.request_session():
        t1 = time.time()
        text, detected_language, used_model_id, used_size = svc.transcribe(
            wav=wav,
            sample_rate=sr,
            model_size=model_size,
            language=normalized_language,
            context=context,
            cpu_mode=cpu_mode,
        )
        t2 = time.time()

    total_ms = int((t2 - t0) * 1000)
    transcribe_ms = int((t2 - t1) * 1000)
    print(json.dumps({
        "event": "asr_timing",
        "model_size": used_size,
        "model_id": used_model_id,
        "language": detected_language or normalized_language or "Auto",
        "duration_s": duration_s,
        "chars": len(text),
        "t_total_ms": total_ms,
        "t_transcribe_ms": transcribe_ms,
    }), flush=True)

    return {
        "text": text,
        "language": detected_language or normalized_language or "",
        "model_id": used_model_id,
        "model_size": used_size,
        "duration_s": duration_s,
        "t_total_ms": total_ms,
        "t_transcribe_ms": transcribe_ms,
    }


@app.post("/api/asr")
def asr(
    file: UploadFile = File(...),
    language: Optional[str] = Form(default=None),
    model_size: str = Form(default="fast"),
    context: str = Form(default=""),
    cpu_mode: bool = Form(default=False),
):
    """Transcribe an uploaded audio file with Qwen3-ASR."""
    return _run_asr_request(
        file=file,
        model_size=model_size,
        language=language,
        context=context,
        cpu_mode=cpu_mode,
    )


@app.post("/v1/audio/transcriptions")
def openai_audio_transcriptions(
    file: UploadFile = File(...),
    model: Optional[str] = Form(default=None),
    language: Optional[str] = Form(default=None),
    prompt: Optional[str] = Form(default=None),
    response_format: str = Form(default="json"),
    cpu_mode: bool = Form(default=False),
):
    """OpenAI Whisper-compatible transcription endpoint backed by Qwen3-ASR."""
    fmt = (response_format or "json").strip().lower()
    if fmt not in {"json", "text", "verbose_json"}:
        raise HTTPException(
            status_code=400,
            detail="response_format must be json, text, or verbose_json",
        )
    result = _run_asr_request(
        file=file,
        model_size=_resolve_openai_asr_model_size(model),
        language=language,
        context=prompt or "",
        cpu_mode=cpu_mode,
    )
    if fmt == "text":
        return Response(content=result["text"], media_type="text/plain; charset=utf-8")
    if fmt == "verbose_json":
        return {
            "task": "transcribe",
            "language": result["language"],
            "duration": result["duration_s"],
            "text": result["text"],
            "model": result["model_id"],
        }
    return {"text": result["text"]}
