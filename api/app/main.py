import base64
import contextlib
import io
import json
import os
import re
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
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from qwen_tts import Qwen3TTSModel
from transformers import LogitsProcessor, LogitsProcessorList

SUPPORTED_LANGUAGES = [
    "Auto", "English", "Chinese", "Japanese", "Korean",
    "French", "German", "Spanish", "Italian", "Portuguese", "Arabic", "Russian",
]
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
model_status = {"state": "idle", "model_id": None, "device": None}  # idle | loading | ready
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
    model_size: str = Field(default="fast", description="fast | quality | auto")
    reference_audio: Optional[str] = Field(default=None, description="Filename from /reference-audio when mode=clone")
    voice_description: Optional[str] = Field(default=None, description="Plain-text voice description when mode=design")
    voice_preset: Optional[str] = Field(default=None, description="Named voice preset to resolve into voice_description")
    user: Optional[str] = Field(default="default", description="User namespace for reference audio")
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
        self.auto_quality_min_vram_mb = int(os.getenv("QWEN_AUTO_QUALITY_MIN_VRAM_MB", "3000"))
        self.max_new_tokens = int(os.getenv("QWEN_MAX_NEW_TOKENS", "192"))
        self.max_new_tokens_per_char = float(os.getenv("QWEN_MAX_NEW_TOKENS_PER_CHAR", "1.0"))
        self.max_new_tokens_buffer = int(os.getenv("QWEN_MAX_NEW_TOKENS_BUFFER", "48"))
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
        self.lock = threading.Lock()
        self.load_lock = threading.Lock()
        self.generation_lock = threading.Lock()
        self._model = None
        self._model_id: Optional[str] = None
        self._model_device: Optional[str] = None

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

    def _unload_model_locked(self):
        if self._model is not None:
            del self._model
            self._model = None
        self._model_id = None
        self._model_device = None
        cleanup_generation_memory()

    def get_model(self, model_id: str, cpu_mode: bool = False):
        model_device = "cpu" if cpu_mode else self.device.strip().lower()
        if self._model is not None and self._model_id == model_id and self._model_device == model_device:
            return self._model

        if not self.load_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="Model load already in progress")

        try:
            with self.lock:
                if self._model is not None and self._model_id == model_id and self._model_device == model_device:
                    return self._model

                model_status.update({"state": "loading", "model_id": model_id, "device": model_device})
                self._unload_model_locked()

                try:
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
                    if model_device.startswith("cuda") and is_cuda_out_of_memory(e):
                        print(json.dumps({
                            "event": "tts_model_load_gpu_busy",
                            "model_id": model_id,
                            "device": model_device,
                            "detail": str(e),
                            "cuda": cuda_memory_snapshot(),
                        }), flush=True)
                        raise gpu_busy_error() from None
                    raise

                self._model_id = model_id
                self._model_device = model_device
                model_status.update({"state": "ready", "model_id": model_id, "device": model_device})
                return self._model
        except Exception:
            model_status.update({"state": "idle", "model_id": None, "device": None})
            raise
        finally:
            self.load_lock.release()

    def unload_model(self) -> None:
        """Drop the loaded model and release its CUDA allocations.

        Wait for an in-flight chunk to finish before removing the service's
        final model reference. TTS endpoints call this once their complete
        request (including all chunks) has finished.
        """
        with self.generation_lock:
            with self.lock:
                unloaded_model_id = self._model_id
                had_model = self._model is not None
                self._unload_model_locked()
                model_status.update({"state": "idle", "model_id": None, "device": None})

        if had_model:
            print(json.dumps({
                "event": "tts_model_unloaded",
                "model_id": unloaded_model_id,
                "cuda": cuda_memory_snapshot(),
            }), flush=True)

    def hard_reset(self):
        self.unload_model()

    def resolve_reference_audio(self, filename: str, user_dir: Path) -> Path:
        candidate = (user_dir / filename).resolve()
        if not str(candidate).startswith(str(user_dir.resolve())):
            raise HTTPException(status_code=400, detail="Invalid filename")
        if not candidate.is_file():
            raise HTTPException(status_code=404, detail=f"Reference audio not found: {filename}")
        return candidate

    def _resolve_model_size(self, mode: str, requested_size: str, cpu_mode: bool = False) -> str:
        size = (requested_size or "quality").strip().lower()
        if size not in {"fast", "quality", "auto"}:
            size = "quality"

        if size == "auto":
            threshold_bytes = self.auto_quality_min_vram_mb * 1024 * 1024
            free_vram_bytes = 0
            total_vram_bytes = 0

            if not cpu_mode and torch.cuda.is_available():
                try:
                    free_vram_bytes, total_vram_bytes = torch.cuda.mem_get_info()
                except Exception:
                    free_vram_bytes = 0
                    total_vram_bytes = 0

            # CUDA availability is irrelevant to a CPU-only request. Prefer
            # the smaller model there (design is normalized to quality below).
            selected_size = "fast" if cpu_mode else ("quality" if free_vram_bytes >= threshold_bytes else "fast")
            print(
                json.dumps(
                    {
                        "event": "tts_model_size_auto",
                        "mode": mode,
                        "requested": "auto",
                        "selected": selected_size,
                        "free_vram_mb": int(free_vram_bytes / (1024 * 1024)),
                        "total_vram_mb": int(total_vram_bytes / (1024 * 1024)),
                        "threshold_mb": self.auto_quality_min_vram_mb,
                    }
                )
            )
            size = selected_size

        if mode == "design" and size == "fast":
            print(json.dumps({"event": "tts_warn", "detail": "design mode does not support fast; using quality"}))
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
        extra encode path.
        """
        mode = (req.mode or "clone").strip().lower()
        if mode != "clone":
            return None
        if not req.reference_audio:
            raise HTTPException(status_code=400, detail="reference_audio required for clone mode")

        size = self._resolve_model_size(mode, req.model_size or "quality", req.cpu_mode)
        if size not in MODEL_IDS[mode]:
            size = "quality"
        model = self.get_model(MODEL_IDS[mode][size], req.cpu_mode)
        ref_path = self.resolve_reference_audio(req.reference_audio, user_dir)

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

                speaker_encoder = model.model.speaker_encoder
                if speaker_encoder is None:
                    raise RuntimeError("loaded model has no speaker_encoder")
                try:
                    spk_param = next(speaker_encoder.parameters())
                    spk_device = spk_param.device
                    spk_dtype = spk_param.dtype
                except StopIteration:
                    spk_device = torch.device("cpu")
                    spk_dtype = torch.float32

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
                cleanup_generation_memory()
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

        size = self._resolve_model_size(mode, req.model_size or "quality", req.cpu_mode)
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

        size = self._resolve_model_size(mode, req.model_size or "quality", req.cpu_mode)

        if size not in MODEL_IDS[mode]:
            size = "quality"

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
        self.state[id(m)] = x[..., x.shape[-1] - m.padding:]
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
            if svc.loaded_model() is not None:
                continue
            if torch.cuda.memory_reserved() < IDLE_CUDA_RELEASE_MIN_MB * 1024 * 1024:
                continue
            # Skip while a model load is in progress; it will manage memory itself.
            if not svc.load_lock.acquire(blocking=False):
                continue
            try:
                if svc.loaded_model() is not None:
                    continue
                before_mb = int(torch.cuda.memory_reserved() / (1024 * 1024))
                cleanup_generation_memory()
            finally:
                svc.load_lock.release()
            print(json.dumps({
                "event": "tts_idle_cuda_released",
                "reserved_before_mb": before_mb,
                "cuda": cuda_memory_snapshot(),
            }), flush=True)
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


@app.get("/status")
def status():
    return {
        "model_state": model_status["state"],
        "model_id": model_status["model_id"],
        "device": model_status["device"],
        "uptime_s": int(time.time() - start_time),
        "cuda": cuda_memory_snapshot(),
    }


@app.get("/readyz")
def readyz():
    if model_status["state"] == "ready":
        return {"ready": True, "model_id": model_status["model_id"], "device": model_status["device"]}
    return JSONResponse(
        status_code=503,
        content={"ready": False, "reason": model_status["state"]},
    )


@app.get("/info")
def info():
    return {
        "models": MODEL_IDS,
        "modes": ["clone", "design"],
        "sizes": ["fast", "quality", "auto"],
        "current_model": svc._model_id,
        "current_device": svc._model_device,
        "supported_languages": SUPPORTED_LANGUAGES,
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

    chunks = chunk_text(req.text)
    n_chunks = len(chunks)
    t0 = time.time()
    try:
        base_extra_kwargs = svc.prepare_request_extra_kwargs(req, user_dir)
        model = svc.loaded_model()
        speech_tokenizer = model.model.speech_tokenizer
        if not QWEN_MEMORY_PATCH_ENABLED:
            raise HTTPException(status_code=501, detail="frame streaming unavailable: talker patch not active")
        if speech_tokenizer.get_model_type() != "qwen3_tts_tokenizer_12hz":
            raise HTTPException(status_code=501, detail="frame streaming requires a 12Hz model")
        sample_rate = int(speech_tokenizer.get_output_sample_rate())
    except Exception:
        svc.unload_model()
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
                cleanup_generation_memory()
                events.put(("chunk_end", i))
            events.put(("done", None))
        except StreamCancelled:
            cleanup_generation_memory()
            events.put(("done", None))
        except Exception as e:
            cleanup_generation_memory()
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
                        "event": "tts_audio_stream_chunk",
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

    threading.Thread(target=run_generation, daemon=True).start()
    payloads = pcm_payloads()

    # Wait for the first audio before sending headers, so a failure (GPU
    # busy, bad voice, ...) is a proper error status, not an empty 200.
    try:
        first = next(payloads, b"")
    except Exception as e:
        cancel.set()
        print(json.dumps({
            "event": "tts_audio_stream_first_audio_error",
            "detail": str(getattr(e, "detail", e)),
            "chunks": n_chunks,
            "cuda": cuda_memory_snapshot(),
        }), flush=True)
        svc.unload_model()
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"tts audio stream failed before first audio: {e}")

    def body():
        try:
            if audio_format == "wav":
                yield wav_stream_header(sample_rate)
            yield first
            yield from payloads
            print(json.dumps({
                "event": "tts_audio_stream_done",
                "mode": req.mode,
                "chars": len(req.text or ""),
                "chunks": n_chunks,
                "frames": stats["frames"],
                "audio_ms": stats["samples"] * 1000 // sample_rate,
                "t_first_audio_ms": stats["t_first_audio_ms"],
                "t_total_ms": int((time.time() - t0) * 1000),
                "cuda": cuda_memory_snapshot(),
            }), flush=True)
        except Exception as e:
            print(json.dumps({
                "event": "tts_audio_stream_error",
                "detail": str(getattr(e, "detail", e)),
                "chunks": n_chunks,
                "cuda": cuda_memory_snapshot(),
            }), flush=True)
            raise
        finally:
            # Normal end, error, or client disconnect: stop generation at the
            # next frame, then release the model. unload_model() waits on
            # generation_lock, so the worker has left generate() by then.
            cancel.set()
            svc.unload_model()

    return StreamingResponse(
        body(),
        media_type="audio/wav" if audio_format == "wav" else "audio/pcm",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Sample-Rate": str(sample_rate),
            "X-Channels": "1",
            "X-Sample-Format": "s16le",
            **({"Content-Disposition": 'inline; filename="tts-stream.wav"'} if audio_format == "wav" else {}),
        },
    )


@app.post("/api/tts/stream")
def tts_stream(req: TTSRequest):
    req = apply_voice_preset(req)
    user_dir = user_audio_dir(req.user or "default")
    audio_format = req.audio_format.lower().strip()
    if audio_format != "wav":
        raise HTTPException(status_code=400, detail="audio_format must be wav for stream endpoint")

    chunks = chunk_text(req.text)
    n_chunks = len(chunks)
    try:
        base_extra_kwargs = svc.prepare_request_extra_kwargs(req, user_dir)
    except Exception:
        svc.unload_model()
        raise

    q: queue.Queue = queue.Queue()

    def run_generation():
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
                cleanup_generation_memory()
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
            detail = e.detail if isinstance(e, HTTPException) else str(e)
            payload = {"detail": detail}
            if isinstance(e, HTTPException):
                payload["status_code"] = e.status_code
            q.put(("error", payload))
        finally:
            svc.unload_model()

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

    chunks = chunk_text(req.text)
    n_chunks = len(chunks)
    try:
        base_extra_kwargs = svc.prepare_request_extra_kwargs(req, user_dir)

        t1 = time.time()
        wav_arrays: list[np.ndarray] = []
        sr_final: int = 24000
        used_model_id: str = ""
        used_size: str = ""

        for i, chunk in enumerate(chunks):
            chunk_req = req.model_copy(update={"text": chunk})
            wav, sr, used_model_id, used_size = svc.synthesize(
                chunk_req, user_dir, extra_kwargs=base_extra_kwargs
            )
            wav = postprocess_chunk_audio(wav, int(sr))
            wav_arrays.append(np.asarray(wav, dtype=np.float32))
            del wav
            cleanup_generation_memory()
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
    finally:
        svc.unload_model()
