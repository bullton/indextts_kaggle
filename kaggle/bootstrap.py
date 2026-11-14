import glob
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

WORK = Path("/kaggle/working")
OUTPUT_ROOT = WORK / "output"
SCRATCH = Path("/tmp/indextts-work")
CLONE = SCRATCH / "index-tts"

DEFAULT_REPO_URL = "https://github.com/index-tts/index-tts.git"
DEFAULT_REPO_REF = "v2.0.0"
DEFAULT_MODEL_REPO = "IndexTeam/IndexTTS-2"


def log(message):
    print(f"[indextts:bootstrap] {message}", flush=True)


def run(command, cwd=None, env=None, check=True):
    log("$ " + " ".join(str(part) for part in command))
    return subprocess.run(
        [str(part) for part in command],
        cwd=str(cwd) if cwd else None,
        env=env,
        check=check,
    )


def write_result(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("RESULT_JSON " + json.dumps(payload, ensure_ascii=False), flush=True)


def find_request():
    paths = sorted(glob.glob("/kaggle/input/*/indextts_request.json"))
    if not paths:
        paths = sorted(glob.glob("/kaggle/input/**/indextts_request.json", recursive=True))
    if not paths:
        raise FileNotFoundError("indextts_request.json was not found under /kaggle/input")
    return Path(paths[0])


REQUIRED_MODEL_FILES = ("config.yaml", "gpt.pth", "s2mel.pth", "bpe.model")
REQUIRED_MODEL_DIRS = ("qwen0.6bemo4-merge",)


def find_model_dir(root):
    root = Path(root)
    if (root / "config.yaml").is_file():
        return root
    matches = sorted(root.glob("**/config.yaml"))
    if matches:
        return matches[0].parent
    return None


def model_ready(root):
    found = find_model_dir(root)
    if found is None:
        return None
    missing = [name for name in REQUIRED_MODEL_FILES if not (found / name).is_file()]
    missing += [name for name in REQUIRED_MODEL_DIRS if not (found / name).is_dir()]
    if missing:
        log(f"model at {found} is incomplete, missing: {', '.join(missing)}")
        return None
    return found


def main():
    started = time.time()
    request = None
    job_id = f"job-{int(time.time())}"
    result_path = OUTPUT_ROOT / job_id / "result.json"
    stage = "prepare"

    try:
        request_path = find_request()
        request = json.loads(request_path.read_text(encoding="utf-8"))
        job_id = request.get("job_id") or job_id
        output_name = request.get("output_name") or "audio.wav"
        if not output_name.lower().endswith(".wav"):
            output_name += ".wav"
        output_dir = OUTPUT_ROOT / job_id
        output_path = output_dir / output_name
        result_path = output_dir / "result.json"
        output_dir.mkdir(parents=True, exist_ok=True)

        SCRATCH.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["GIT_LFS_SKIP_SMUDGE"] = "1"
        env["UV_CACHE_DIR"] = str(SCRATCH / "uv-cache")
        env["UV_LINK_MODE"] = "copy"

        repo_url = request.get("repo_url") or DEFAULT_REPO_URL
        repo_ref = request.get("repo_ref") or DEFAULT_REPO_REF
        model_repo = request.get("model_repo") or DEFAULT_MODEL_REPO

        stage = "clone"
        if not (CLONE / "pyproject.toml").is_file():
            run(
                ["git", "clone", "--depth", "1", "--branch", repo_ref, repo_url, str(CLONE)],
                env=env,
            )
        else:
            log(f"repository already present at {CLONE}")

        stage = "dependencies"
        log("installing IndexTTS dependencies with uv (several GB on the first run)")
        run(["uv", "sync", "--frozen"], cwd=CLONE, env=env)
        venv_python = CLONE / ".venv" / "bin" / "python"
        if not venv_python.is_file():
            raise RuntimeError(f"uv sync did not create {venv_python}")

        stage = "model"
        model_dir = None
        if request.get("model_dir"):
            model_dir = find_model_dir(request["model_dir"])
            if model_dir is None:
                raise FileNotFoundError(
                    f"model_dir {request['model_dir']} does not contain config.yaml"
                )
            log(f"using model dataset at {model_dir}")
        else:
            model_dir = CLONE / "checkpoints"
            if model_ready(model_dir) is None:
                model_dir.mkdir(parents=True, exist_ok=True)
                log(f"downloading {model_repo} to {model_dir}")
                download_env = env.copy()
                download_env["HF_HOME"] = str(SCRATCH / "hf-home")
                download_env["HF_HUB_CACHE"] = str(SCRATCH / "hf-cache")
                code = (
                    "from huggingface_hub import snapshot_download;"
                    f"snapshot_download(repo_id={model_repo!r}, local_dir={str(model_dir)!r}, max_workers=8)"
                )
                run([str(venv_python), "-c", code], cwd=CLONE, env=download_env)
            model_dir = model_ready(model_dir)
            if model_dir is None:
                raise RuntimeError("model download did not produce a complete checkpoint directory")

        stage = "worker"
        worker_path = SCRATCH / "worker.py"
        worker_path.write_text(WORKER_SOURCE, encoding="utf-8")
        worker_env = env.copy()
        worker_env["INDEXTTS_REQUEST_PATH"] = str(request_path)
        worker_env["INDEXTTS_RESULT_PATH"] = str(result_path)
        worker_env["INDEXTTS_JOB_ID"] = job_id
        worker_env["INDEXTTS_OUTPUT_PATH"] = str(output_path)
        worker_env["INDEXTTS_MODEL_DIR"] = str(model_dir)
        worker_env["INDEXTTS_CLONE_DIR"] = str(CLONE)
        worker_env["PYTHONPATH"] = str(CLONE) + os.pathsep + worker_env.get("PYTHONPATH", "")
        log(f"starting inference worker for job {job_id}")
        return_code = subprocess.call([str(venv_python), str(worker_path)], cwd=str(CLONE), env=worker_env)

        if not result_path.is_file():
            write_result(
                result_path,
                {
                    "status": "error",
                    "stage": "worker",
                    "job_id": job_id,
                    "error": f"worker exited with code {return_code} without writing a result",
                    "elapsed_sec": round(time.time() - started, 2),
                },
            )
        log(f"kernel finished with code {return_code}; total {round(time.time() - started, 2)}s")
        sys.exit(return_code)
    except Exception as exc:
        traceback.print_exc()
        payload = {
            "status": "error",
            "stage": stage,
            "job_id": job_id,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "elapsed_sec": round(time.time() - started, 2),
        }
        try:
            write_result(result_path, payload)
        except Exception:
            write_result(OUTPUT_ROOT / job_id / "result.json", payload)
        sys.exit(1)


if __name__ == "__main__":
    main()
