import glob
import json
import os
import sys
import time
import traceback
from pathlib import Path

OUTPUT_ROOT = Path("/kaggle/working/output")


def log(message):
    print(f"[indextts:worker] {message}", flush=True)


def write_result(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("RESULT_JSON " + json.dumps(payload, ensure_ascii=False), flush=True)


def resolve_input(raw, label):
    if not raw:
        return None
    path = Path(raw)
    if path.is_file():
        return str(path.resolve())

    name = path.name
    patterns = []
    parts = path.parts
    if "input" in parts:
        tail = parts[parts.index("input") + 1:]
        if len(tail) >= 2:
            slug = tail[0]
            rest = "/".join(tail[1:])
            patterns.append(f"/kaggle/input/datasets/*/{slug}/{rest}")
            patterns.append(f"/kaggle/input/{slug}/{rest}")
    patterns.append(f"/kaggle/input/datasets/*/*/{name}")
    patterns.append(f"/kaggle/input/*/{name}")

    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            return str(Path(matches[0]).resolve())
    raise FileNotFoundError(f"{label} not found: {raw}")


def parse_vector(value):
    if value is None or value == "":
        return None
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list) or len(value) != 8:
        raise ValueError("emo_vector must be a list of 8 numbers")
    return [float(item) for item in value]


def audio_metadata(path):
    try:
        import soundfile as sf

        info = sf.info(path)
        return {
            "duration_sec": round(float(info.duration), 3),
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
        }
    except Exception:
        return {}


def main():
    request_path = Path(os.environ["INDEXTTS_REQUEST_PATH"])
    result_path = Path(os.environ["INDEXTTS_RESULT_PATH"])
    job_id = os.environ.get("INDEXTTS_JOB_ID", "unknown")
    output_path = Path(os.environ["INDEXTTS_OUTPUT_PATH"])
    model_dir = Path(os.environ["INDEXTTS_MODEL_DIR"])
    started = time.time()

    request = json.loads(request_path.read_text(encoding="utf-8"))
    text = (request.get("text") or "").strip()
    if not text:
        raise ValueError("request.text is empty")

    voice_path = resolve_input(request.get("voice_path"), "voice reference")
    emo_audio_path = resolve_input(request.get("emo_audio_path"), "emotion reference")

    os.chdir(os.environ.get("INDEXTTS_CLONE_DIR", "/tmp/indextts-work/index-tts"))
    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, os.environ.get("INDEXTTS_CLONE_DIR", os.getcwd()))

    import torch
    from indextts.infer_v2 import IndexTTS2

    log(f"torch {torch.__version__}; cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"gpu {torch.cuda.get_device_name(0)}")

    tts = IndexTTS2(
        cfg_path=str(model_dir / "config.yaml"),
        model_dir=str(model_dir),
        use_fp16=bool(request.get("use_fp16", True)),
        use_cuda_kernel=False,
        use_deepspeed=False,
        use_torch_compile=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    infer_kwargs = {
        "spk_audio_prompt": voice_path,
        "text": text,
        "output_path": str(output_path),
        "verbose": True,
        "max_text_tokens_per_segment": int(request.get("max_text_tokens_per_segment") or 120),
    }
    if emo_audio_path:
        infer_kwargs["emo_audio_prompt"] = emo_audio_path
        infer_kwargs["emo_alpha"] = float(request.get("emo_alpha", 1.0))
    emo_vector = parse_vector(request.get("emo_vector"))
    if emo_vector is not None:
        infer_kwargs["emo_vector"] = emo_vector
    if request.get("emo_text"):
        infer_kwargs["emo_text"] = request["emo_text"]
    if request.get("use_emo_text"):
        infer_kwargs["use_emo_text"] = True
        infer_kwargs["emo_alpha"] = float(request.get("emo_alpha", 0.6))
    if request.get("use_random") is not None:
        infer_kwargs["use_random"] = bool(request["use_random"])
    if request.get("interval_silence") is not None:
        infer_kwargs["interval_silence"] = int(request["interval_silence"])

    log(f"inference start; text_chars={len(text)}; voice={voice_path}")
    tts.infer(**infer_kwargs)

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"inference finished without producing {output_path}")

    meta = audio_metadata(str(output_path))
    result = {
        "status": "ok",
        "job_id": job_id,
        "audio_path": str(output_path),
        "audio_file": str(output_path.relative_to("/kaggle/working")),
        "audio_bytes": output_path.stat().st_size,
        "text_chars": len(text),
        "voice_path": voice_path,
        "model_dir": str(model_dir),
        "elapsed_sec": round(time.time() - started, 2),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        **meta,
    }
    write_result(result_path, result)
    log(f"done in {result['elapsed_sec']}s -> {output_path}")


if __name__ == "__main__":
    request_file = Path(os.environ["INDEXTTS_REQUEST_PATH"])
    result_file = Path(os.environ["INDEXTTS_RESULT_PATH"])
    job = os.environ.get("INDEXTTS_JOB_ID", "unknown")
    started_at = time.time()
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        payload = {
            "status": "error",
            "stage": "inference",
            "job_id": job,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "elapsed_sec": round(time.time() - started_at, 2),
            "request_path": str(request_file),
        }
        try:
            write_result(result_file, payload)
        except Exception:
            fallback = OUTPUT_ROOT / job / "result.json"
            write_result(fallback, payload)
        sys.exit(1)
