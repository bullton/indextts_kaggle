# IndexTTS2 × Kaggle API 语音克隆流水线

把原来 Colab 里的 IndexTTS2 部署改造成可编程调用的 Kaggle 任务：

1. 本地脚本把文稿 JSON 和音色样本写入一个私有 Kaggle Dataset。
2. 通过 Kaggle API 推送并启动一个 GPU Kernel。
3. Kernel 内自动克隆仓库、安装依赖、下载模型、执行语音克隆。
4. 本地脚本轮询状态，下载 `/kaggle/working/output/<job_id>/audio.wav` 并返回本地路径。

Kaggle 本身不是常驻服务，这里的“自动起实例”对应 Kaggle 的 **push 一个 Kernel 版本并启动运行**。每次调用都会排队等 GPU，运行结束后实例自动释放。

## 目录

```
kaggle/
  bootstrap.py        # 在 Kaggle 系统 Python 中运行：装环境、下模型、启动 worker
  worker.py           # 在项目 uv 虚拟环境中运行：加载 IndexTTS2 并合成音频
  run_kaggle_tts.py   # 本地 CLI：上传数据集、推送 Kernel、轮询、下载结果
  examples/
    request.example.json
```

`indexTTS.ipynb` 保持原样，作为 Colab 版本参考。

## 前置条件

- 已安装 `kaggle` CLI（本仓库使用项目内 `.venv`）：

```bash
uv venv .venv
uv pip install --python .venv/bin/python "kaggle>=2.2.5" six
```

- Kaggle 凭据放在 `~/.kaggle/access_token`（或设置 `KAGGLE_API_TOKEN`）。
- Kaggle 账号需要开启 GPU 和 Internet（免费 GPU 额度每周 30 小时）。

CLI 会自动使用真实用户目录下的 `~/.kaggle`，即使当前 shell 的 `HOME` 被其他工具改写。

## 快速开始

用本地音色样本合成一句话：

```bash
.venv/bin/python kaggle/run_kaggle_tts.py \
  --text "你好，这是通过 Kaggle API 运行的 IndexTTS2 语音克隆测试。" \
  --voice voice/0328.wav
```

首次运行会：

- 创建私有 Dataset ` <username>/indextts-request`
- 创建私有 Kernel `<username>/indextts-clone`
- 下载约 5.9 GB 的 IndexTTS2 权重到实例的 `/tmp`（不占输出配额）
- 合成后把音频下载到 `outputs/<job_id>/output/<job_id>/audio.wav`

## 使用 JSON 文稿

```bash
.venv/bin/python kaggle/run_kaggle_tts.py --request kaggle/examples/request.example.json
```

`--request` 支持这些字段：

```json
{
  "text": "要合成的文稿",
  "voice_path": "/kaggle/input/indextts-request/voice.wav",
  "emo_alpha": 1.0,
  "emo_vector": [0, 0, 0, 0, 0, 0, 0.4, 0],
  "emo_text": "平静但略带惊讶",
  "use_emo_text": false,
  "use_random": false,
  "max_text_tokens_per_segment": 120,
  "use_fp16": true,
  "output_name": "audio.wav",
  "repo_ref": "v2.0.0",
  "model_repo": "IndexTeam/IndexTTS-2"
}
```

命令行参数会覆盖 JSON 中的同名字段。

## 音色已经存在 Kaggle

如果音色样本已经上传为私有 Dataset，例如 `yourname/my-voices`，其中文件为 `samples/me.wav`：

```bash
.venv/bin/python kaggle/run_kaggle_tts.py \
  --text "直接使用 Kaggle 上已有的音色。" \
  --voice-dataset yourname/my-voices \
  --voice-file samples/me.wav
```

这样不会重复上传音色文件，Kernel 会挂载该 Dataset。

## 情绪控制

```bash
# 情感参考音频
.venv/bin/python kaggle/run_kaggle_tts.py \
  --text "你吓死我了！" \
  --voice voice/0328.wav \
  --emo-audio path/to/angry.wav \
  --emo-alpha 0.9

# 8 维情感向量：[开心, 生气, 悲伤, 害怕, 厌恶, 忧郁, 惊讶, 平静]
.venv/bin/python kaggle/run_kaggle_tts.py \
  --text "这也太离谱了。" \
  --voice voice/0328.wav \
  --emo-vector "[0,0,0,0,0,0,0.5,0]"
```

## 输出

成功时 CLI 会打印 JSON 摘要，关键字段：

```json
{
  "status": "ok",
  "job_id": "20261114-153000-ab12cd",
  "audio_path": "/.../outputs/20261114-153000-ab12cd/output/20261114-153000-ab12cd/audio.wav",
  "duration_sec": 8.42,
  "kernel_url": "https://www.kaggle.com/code/yourname/indextts-clone",
  "kernel_status": "COMPLETE"
}
```

同目录下还有：

- `result.json`：Kernel 内生成的音频元数据
- `kernel.log`：运行日志

## 模型缓存（可选，推荐长期使用）

每次运行都从 HuggingFace 下载 5.9 GB 模型会浪费时间。可以把一次下载好的 `checkpoints/` 目录上传为私有 Dataset，例如 `yourname/indextts2-checkpoints`，然后：

```bash
.venv/bin/python kaggle/run_kaggle_tts.py \
  --text "使用缓存的模型。" \
  --voice voice/0328.wav \
  --model-dataset yourname/indextts2-checkpoints
```

Kernel 会直接在 `/kaggle/input/<slug>` 中查找 `config.yaml`，跳过下载。

## 调试

只生成 Dataset 和 Kernel 文件、不调用 Kaggle：

```bash
.venv/bin/python kaggle/run_kaggle_tts.py \
  --text "调试" --voice voice/0328.wav --dry-run
```

生成物在 `.kaggle_jobs/<job_id>/`：

- `dataset/indextts_request.json`
- `dataset/voice.wav`
- `kernel/main.py`（由 `bootstrap.py` + `worker.py` 自动拼接）
- `kernel/kernel-metadata.json`

只推送、不等待：

```bash
.venv/bin/python kaggle/run_kaggle_tts.py \
  --text "后台运行" --voice voice/0328.wav --no-wait
```

然后到 `https://www.kaggle.com/code/<username>/indextts-clone` 查看运行页。

如果本地等待进程被中断，但 Kaggle 上的任务还在跑，可以用 job_id 继续等待并下载：

```bash
.venv/bin/python kaggle/run_kaggle_tts.py --resume 20261114-153019-1303f3
```

指定某个 Kernel 版本下载：

```bash
.venv/bin/python kaggle/run_kaggle_tts.py --resume 20261114-153019-1303f3 --kernel-version 1
```

## 重要限制

- **不是实时 API**：Kaggle 任务是批处理，免费 GPU 队列可能需要 30–90 分钟甚至更久；单次最长 12 小时，CLI 默认最多等待 3 小时（`--timeout`）。
- **免费 GPU 配额**：每周 30 小时，调试次数要节制。
- **依赖联网**：Kernel 需要 `enable_internet: true` 才能 clone 仓库和下载模型。
- **模型版本**：默认固定 `index-tts` 的 `v2.0.0` 标签，对应 `checkpoints/config.yaml` 布局；不要随意改成 `main`，当前 `main` 已改用 `checkpoints_2` 布局。
- **参考音频**：IndexTTS2 只取参考音频前 15 秒，建议使用 5–15 秒、干净、单人说话的片段。
- **并发**：请求 Dataset 使用固定 slug，多个任务同时运行会互相覆盖，建议串行调用。
- **私密性**：Dataset 和 Kernel 默认都是私有的，音色和文稿不会公开。
