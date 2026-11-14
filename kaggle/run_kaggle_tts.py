#!/usr/bin/env python3
import argparse
import json
import os
import pwd
import re
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DEFAULT_KERNEL_SLUG = "indextts-clone"
DEFAULT_REQUEST_SLUG = "indextts-request"
DEFAULT_REPO_URL = "https://github.com/index-tts/index-tts.git"
DEFAULT_REPO_REF = "v2.0.0"
DEFAULT_MODEL_REPO = "IndexTeam/IndexTTS-2"
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}


def log(message):
    print(f"[kaggle-tts] {message}", flush=True)


def fail(message, code=2):
    print(f"[kaggle-tts] ERROR: {message}", file=sys.stderr, flush=True)
    sys.exit(code)


def real_home():
    return Path(pwd.getpwuid(os.getuid()).pw_dir)


def find_kaggle_bin(explicit=None):
    candidates = [
        explicit,
        os.environ.get("KAGGLE_BIN"),
        shutil.which("kaggle"),
        str(PROJECT_ROOT / ".venv" / "bin" / "kaggle"),
        str(real_home() / ".local" / "bin" / "kaggle"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    fail(
        "未找到 kaggle CLI。请先运行：\n"
        "  uv venv .venv\n"
        '  uv pip install --python .venv/bin/python "kaggle>=2.2.5" six'
    )


def build_env():
    env = os.environ.copy()
    home = real_home()
    env["HOME"] = str(home)
    config_dir = Path(env.get("KAGGLE_CONFIG_DIR") or (home / ".kaggle"))
    env["KAGGLE_CONFIG_DIR"] = str(config_dir)
    has_token = bool(env.get("KAGGLE_API_TOKEN") or env.get("KAGGLE_KEY"))
    has_file = (config_dir / "access_token").is_file() or (config_dir / "kaggle.json").is_file()
    if not (has_token or has_file):
        fail(
            f"未找到 Kaggle 凭据。请确认 {config_dir / 'access_token'} 存在，"
            "或设置 KAGGLE_API_TOKEN。"
        )
    return env


def kaggle(kaggle_bin, env, args, check=True):
    command = [kaggle_bin] + [str(item) for item in args]
    result = subprocess.run(command, env=env, text=True, capture_output=True)
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        fail(f"kaggle {' '.join(str(a) for a in args)} 失败：{detail}")
    return result


def get_username(kaggle_bin, env, explicit=None):
    if explicit:
        return explicit
    result = kaggle(kaggle_bin, env, ["config", "view"], check=False)
    match = re.search(r"username:\s*(\S+)", result.stdout or "")
    if match:
        return match.group(1)
    fail("无法从 Kaggle 配置读取用户名，请通过 --username 指定。")


def slug_of(ref):
    return ref.split("/", 1)[1] if "/" in ref else ref


def parse_version(text):
    match = re.search(r"Kernel version\s+(\d+)", text or "")
    return int(match.group(1)) if match else None


def parse_status(text):
    match = re.search(r'status\s+"([^"]+)"', text or "")
    if match:
        return match.group(1).replace("KernelWorkerStatus.", "")
    return (text or "").strip().splitlines()[-1] if (text or "").strip() else "UNKNOWN"


def load_json(path):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"无法读取 JSON {path}: {exc}")
    if not isinstance(payload, dict):
        fail(f"JSON 顶层必须是对象: {path}")
    return payload


def resolve_text(args, request):
    if args.text and args.text_file:
        fail("--text 和 --text-file 只能二选一。")
    if args.text:
        return args.text.strip()
    if args.text_file:
        return Path(args.text_file).read_text(encoding="utf-8").strip()
    return (request.get("text") or "").strip()


def stage_dataset(args, request, work_dir, request_dataset):
    stage = work_dir / "dataset"
    stage.mkdir(parents=True, exist_ok=True)
    voice = args.voice
    voice_name = None

    if voice:
        voice_path = Path(voice).expanduser().resolve()
        if not voice_path.is_file():
            fail(f"音色文件不存在: {voice_path}")
        if voice_path.suffix.lower() not in AUDIO_SUFFIXES:
            fail(f"不支持的音频格式: {voice_path.suffix}")
        voice_name = f"voice{voice_path.suffix.lower()}"
        shutil.copy2(voice_path, stage / voice_name)
        request["voice_path"] = f"/kaggle/input/{slug_of(request_dataset)}/{voice_name}"
        log(f"音色文件将随请求上传: {voice_path.name} -> {voice_name}")
    elif args.voice_dataset:
        if not args.voice_file:
            fail("使用 --voice-dataset 时必须同时提供 --voice-file。")
        request["voice_path"] = f"/kaggle/input/{slug_of(args.voice_dataset)}/{args.voice_file.lstrip('/')}"
        request["voice_dataset"] = args.voice_dataset
    elif not request.get("voice_path"):
        fail("缺少音色：请提供 --voice、--voice-dataset + --voice-file，或在 --request 中写 voice_path。")

    if args.emo_audio:
        emo_path = Path(args.emo_audio).expanduser().resolve()
        if not emo_path.is_file():
            fail(f"情感参考音频不存在: {emo_path}")
        if emo_path.suffix.lower() not in AUDIO_SUFFIXES:
            fail(f"不支持的情感音频格式: {emo_path.suffix}")
        emo_name = f"emotion{emo_path.suffix.lower()}"
        shutil.copy2(emo_path, stage / emo_name)
        request["emo_audio_path"] = f"/kaggle/input/{slug_of(request_dataset)}/{emo_name}"

    if args.voice_dataset:
        request["voice_dataset"] = args.voice_dataset
    if args.model_dataset:
        request["model_dir"] = f"/kaggle/input/{slug_of(args.model_dataset)}"
        request["model_dataset"] = args.model_dataset
    if args.repo_url:
        request["repo_url"] = args.repo_url
    if args.repo_ref:
        request["repo_ref"] = args.repo_ref
    if args.model_repo:
        request["model_repo"] = args.model_repo
    if args.use_fp16 is not None:
        request["use_fp16"] = bool(args.use_fp16)
    if args.emo_alpha is not None:
        request["emo_alpha"] = float(args.emo_alpha)
    if args.emo_vector:
        vector = json.loads(args.emo_vector)
        if not isinstance(vector, list) or len(vector) != 8:
            fail("--emo-vector 需要 8 个数字，例如 '[0,0,0,0,0,0,0.4,0]'。")
        request["emo_vector"] = [float(item) for item in vector]
    if args.emo_text:
        request["emo_text"] = args.emo_text
    if args.use_emo_text is not None:
        request["use_emo_text"] = bool(args.use_emo_text)
    if args.use_random is not None:
        request["use_random"] = bool(args.use_random)
    if args.max_text_tokens is not None:
        request["max_text_tokens_per_segment"] = int(args.max_text_tokens)
    if args.output_name:
        name = Path(args.output_name).name
        request["output_name"] = name if name.lower().endswith(".wav") else f"{name}.wav"

    (stage / "indextts_request.json").write_text(
        json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "title": "IndexTTS Request",
        "id": request_dataset,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (stage / "dataset-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return stage


def stage_kernel(args, request, work_dir, kernel_ref, dataset_sources):
    kernel_dir = work_dir / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    worker_source = (HERE / "worker.py").read_text(encoding="utf-8")
    bootstrap_source = (HERE / "bootstrap.py").read_text(encoding="utf-8")
    main_source = f"WORKER_SOURCE = {worker_source!r}\n\n{bootstrap_source}"
    (kernel_dir / "main.py").write_text(main_source, encoding="utf-8")

    metadata = {
        "id": kernel_ref,
        "title": args.kernel_title,
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": not args.no_gpu,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": args.accelerator or "",
        "dataset_sources": dataset_sources,
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return kernel_dir


def wait_dataset_ready(kaggle_bin, env, dataset_ref, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = kaggle(kaggle_bin, env, ["datasets", "status", dataset_ref], check=False)
        status = ((result.stdout or "") + (result.stderr or "")).strip().lower()
        if status.startswith("ready"):
            return True
        if "error" in status or "failed" in status:
            fail(f"数据集处理失败: {dataset_ref}: {status}")
        time.sleep(5)
    return False


def publish_dataset(kaggle_bin, env, stage, request_dataset, job_id):
    message = f"{job_id}: IndexTTS request"
    result = kaggle(
        kaggle_bin,
        env,
        ["datasets", "version", "-p", str(stage), "-m", message],
        check=False,
    )
    if result.returncode == 0:
        log(f"请求数据集已更新: {request_dataset}")
        wait_dataset_ready(kaggle_bin, env, request_dataset)
        return "version"
    detail = (result.stderr or result.stdout or "").lower()
    if "not found" in detail or "404" in detail or "does not exist" in detail:
        result = kaggle(kaggle_bin, env, ["datasets", "create", "-p", str(stage)], check=False)
        if result.returncode != 0:
            fail(f"创建请求数据集失败：{(result.stderr or result.stdout).strip()}")
        log(f"请求数据集已创建: {request_dataset}")
        wait_dataset_ready(kaggle_bin, env, request_dataset)
        return "create"
    fail(f"更新请求数据集失败：{(result.stderr or result.stdout).strip()}")


def push_kernel(kaggle_bin, env, kernel_dir, args):
    command = ["kernels", "push", "-p", str(kernel_dir)]
    if args.run_timeout:
        command += ["-t", str(int(args.run_timeout))]
    result = kaggle(kaggle_bin, env, command, check=False)
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        fail(f"推送 kernel 失败：{output.strip()}")
    version = parse_version(output)
    log(output.strip())
    return version


def wait_for_kernel(kaggle_bin, env, kernel_ref, timeout, interval):
    deadline = time.time() + timeout
    last_status = None
    while time.time() < deadline:
        result = kaggle(kaggle_bin, env, ["kernels", "status", kernel_ref], check=False)
        status = parse_status((result.stdout or "") + (result.stderr or ""))
        if status != last_status:
            log(f"状态: {status}")
            last_status = status
        if status == "COMPLETE":
            return status
        if status in {"ERROR", "CANCELED", "CANCEL_ACKNOWLEDGED", "CANCEL_REQUESTED"}:
            return status
        time.sleep(interval)
    return "TIMEOUT"


def resolve_kernel_ref(args, username):
    kernel_ref = args.kernel or f"{username}/{DEFAULT_KERNEL_SLUG}"
    if "/" not in kernel_ref:
        fail("--kernel 必须是 owner/slug 格式。")
    if args.kernel_version and args.resume:
        kernel_ref = f"{kernel_ref}/{args.kernel_version}"
    return kernel_ref


def download_result(kaggle_bin, env, kernel_ref, job_id, out_root):
    job_dir = out_root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    kaggle(kaggle_bin, env, ["kernels", "output", kernel_ref, "-p", str(job_dir), "-o"], check=False)
    log_path = job_dir / "kernel.log"
    logs = kaggle(kaggle_bin, env, ["kernels", "logs", kernel_ref], check=False)
    log_path.write_text((logs.stdout or "") + (logs.stderr or ""), encoding="utf-8")

    result_candidates = sorted(job_dir.glob(f"**/output/{job_id}/result.json"))
    if not result_candidates:
        result_candidates = sorted(job_dir.glob("**/result.json"))
    result = load_json(result_candidates[0]) if result_candidates else None

    audio_path = None
    if result and result.get("audio_file"):
        candidate = job_dir / result["audio_file"]
        if candidate.is_file():
            audio_path = candidate
    if audio_path is None:
        wavs = sorted(job_dir.glob(f"**/output/{job_id}/*.wav")) or sorted(job_dir.glob("**/*.wav"))
        if wavs:
            audio_path = wavs[0]
    return job_dir, log_path, result, audio_path


def build_parser():
    parser = argparse.ArgumentParser(
        description="Push an IndexTTS2 voice-cloning request to Kaggle and download the audio."
    )
    parser.add_argument("--text", help="要合成的文稿")
    parser.add_argument("--text-file", help="从文本文件读取文稿")
    parser.add_argument("--request", help="请求 JSON（可包含 text、voice_path、情感参数等）")
    parser.add_argument("--voice", help="本地音色参考音频")
    parser.add_argument("--voice-dataset", help="已存放在 Kaggle 的音色数据集，如 owner/voices")
    parser.add_argument("--voice-file", help="音色数据集内的相对路径")
    parser.add_argument("--emo-audio", help="本地情感参考音频")
    parser.add_argument("--emo-alpha", type=float)
    parser.add_argument("--emo-vector", help="8 维情感向量 JSON，如 '[0,0,0,0,0,0,0.4,0]'")
    parser.add_argument("--emo-text", help="情感描述文本")
    parser.add_argument("--use-emo-text", dest="use_emo_text", action="store_true", default=None)
    parser.add_argument("--use-random", dest="use_random", action="store_true", default=None)
    parser.add_argument("--max-text-tokens", type=int)
    parser.add_argument("--use-fp16", dest="use_fp16", action="store_true", default=None)
    parser.add_argument("--no-fp16", dest="use_fp16", action="store_false")
    parser.add_argument("--output-name", default="audio.wav")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--repo-ref", default=DEFAULT_REPO_REF)
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--model-dataset", help="已缓存的 IndexTTS2 模型数据集，如 owner/indextts2-checkpoints")
    parser.add_argument("--kernel", help="Kernel 引用，默认 <username>/indextts-clone")
    parser.add_argument("--kernel-title", default="IndexTTS Clone")
    parser.add_argument("--request-dataset", help="请求数据集引用，默认 <username>/indextts-request")
    parser.add_argument("--username")
    parser.add_argument("--accelerator", help="Kaggle 加速器，如 NvidiaTeslaT4")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "outputs"), help="下载目录")
    parser.add_argument("--work-dir", default=str(PROJECT_ROOT / ".kaggle_jobs"))
    parser.add_argument("--job-id")
    parser.add_argument("--resume", metavar="JOB_ID", help="不重新推送，只等待并下载已有任务")
    parser.add_argument("--kernel-version", type=int, help="指定要下载的 kernel 版本")
    parser.add_argument("--timeout", type=int, default=10800, help="等待运行的秒数")
    parser.add_argument("--run-timeout", type=int, help="Kernel 最大运行秒数")
    parser.add_argument("--poll-interval", type=int, default=20)
    parser.add_argument("--no-wait", action="store_true", help="推送后立即返回")
    parser.add_argument("--dry-run", action="store_true", help="只生成请求与 kernel 文件，不调用 Kaggle")
    parser.add_argument("--json-out", help="把最终摘要写入 JSON 文件")
    parser.add_argument("--kaggle-bin")
    return parser


def run_resume(args, kaggle_bin, env, kernel_ref, out_root):
    job_id = args.resume
    log(f"等待已有任务: {job_id} ({kernel_ref})")
    status = wait_for_kernel(kaggle_bin, env, kernel_ref, args.timeout, args.poll_interval)
    log(f"最终状态: {status}")
    job_dir, log_path, result, audio_path = download_result(
        kaggle_bin, env, kernel_ref, job_id, out_root
    )
    ok = status == "COMPLETE" and result and result.get("status") == "ok" and audio_path is not None
    summary = {
        "status": "ok" if ok else "error",
        "job_id": job_id,
        "kernel": kernel_ref,
        "kernel_status": status,
        "kernel_url": f"https://www.kaggle.com/code/{kernel_ref}",
        "log_path": str(log_path),
        "audio_path": str(audio_path) if audio_path else None,
        "duration_sec": result.get("duration_sec") if result else None,
        "error": None if ok else (result or {}).get("error") or f"kernel status {status}",
        "kaggle_result": result,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if ok:
        log(f"音频地址: {audio_path}")
        return
    fail(f"语音克隆失败，状态 {status}。日志: {log_path}", code=1)


def main():
    args = build_parser().parse_args()
    env = build_env()
    kaggle_bin = find_kaggle_bin(args.kaggle_bin)
    username = get_username(kaggle_bin, env, args.username)
    kernel_ref = resolve_kernel_ref(args, username)
    out_root = Path(args.out).expanduser().resolve()

    if args.resume:
        run_resume(args, kaggle_bin, env, kernel_ref, out_root)
        return

    request = load_json(args.request) if args.request else {}
    text = resolve_text(args, request)
    if not text:
        fail("缺少文稿：请使用 --text、--text-file 或 --request 提供 text。")

    request_dataset = args.request_dataset or f"{username}/{DEFAULT_REQUEST_SLUG}"
    if "/" not in request_dataset:
        fail("--request-dataset 必须是 owner/slug 格式。")

    job_id = args.job_id or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    request["job_id"] = job_id
    request["text"] = text

    work_dir = Path(args.work_dir).expanduser().resolve() / job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    stage = stage_dataset(args, request, work_dir, request_dataset)
    dataset_sources = [request_dataset]
    for extra in (request.get("voice_dataset"), request.get("model_dataset")):
        if extra and extra not in dataset_sources:
            dataset_sources.append(extra)
    kernel_dir = stage_kernel(args, request, work_dir, kernel_ref, dataset_sources)

    log(f"job_id: {job_id}")
    log(f"kernel: {kernel_ref}")
    log(f"request dataset: {request_dataset}")
    log(f"staging: {work_dir}")

    if args.dry_run:
        print(json.dumps({"job_id": job_id, "stage": str(work_dir), "text_chars": len(text)}, ensure_ascii=False, indent=2))
        return

    dataset_action = publish_dataset(kaggle_bin, env, stage, request_dataset, job_id)
    version = push_kernel(kaggle_bin, env, kernel_dir, args)
    kernel_url = f"https://www.kaggle.com/code/{kernel_ref}"

    summary = {
        "status": "pushed",
        "job_id": job_id,
        "kernel": kernel_ref,
        "kernel_version": version,
        "kernel_url": kernel_url,
        "request_dataset": request_dataset,
        "dataset_action": dataset_action,
    }
    if args.no_wait:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    status = wait_for_kernel(kaggle_bin, env, kernel_ref, args.timeout, args.poll_interval)
    log(f"最终状态: {status}")
    job_dir, log_path, result, audio_path = download_result(kaggle_bin, env, kernel_ref, job_id, out_root)

    ok = status == "COMPLETE" and result and result.get("status") == "ok" and audio_path is not None
    summary.update(
        {
            "status": "ok" if ok else "error",
            "kernel_status": status,
            "log_path": str(log_path),
            "result_path": str(job_dir / "result.json") if (job_dir / "result.json").is_file() else None,
            "audio_path": str(audio_path) if audio_path else None,
            "duration_sec": result.get("duration_sec") if result else None,
            "error": None if ok else (result or {}).get("error") or f"kernel status {status}",
            "kaggle_result": result,
        }
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if ok:
        log(f"音频地址: {audio_path}")
        log(f"运行页: {kernel_url}")
        return
    fail(f"语音克隆失败，状态 {status}。日志: {log_path}", code=1)


if __name__ == "__main__":
    main()
