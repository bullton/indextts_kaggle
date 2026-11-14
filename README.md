# indextts_kaggle

用 Kaggle 免费 GPU 跑 IndexTTS-2 语音克隆，通过 API 调用：本地推文稿 + 音色样本，Kaggle 自动起实例合成，本地取回音频。

原来在 Colab 里的版本保留为 `indexTTS.ipynb`，仅作参考；实际可用的是 `kaggle/` 下的流水线。

## 工作原理

```
本地 run_kaggle_tts.py
  ├─ 写入私有 Dataset: <user>/indextts-request（indextts_request.json + voice.wav）
  ├─ 推送私有 Kernel:  <user>/indextts-clone（main.py = bootstrap.py + worker.py）
  │     Kaggle 起 GPU 实例 → clone index-tts → uv sync → 下载/挂载模型 → worker 推理
  └─ 轮询状态 → 下载 /kaggle/working/output/<job_id>/audio.wav → outputs/<job_id>/
```

Kaggle 没有常驻服务，"自动起实例" 就是 **push 一个 Kernel 版本并运行**。它是批处理，不是实时 API。

## 目录

```
kaggle/
  run_kaggle_tts.py     # 本地 CLI：组请求、传 Dataset、推 Kernel、轮询、下载
  bootstrap.py          # Kaggle 系统 Python 里跑：clone、装依赖、准备模型、拉起 worker
  worker.py             # 项目 venv 里跑：加载 IndexTTS2，合成音频，写 result.json
  examples/request.example.json
  README.md             # 更细的参数与情绪控制说明
indexTTS.ipynb          # 原 Colab 版本，参考用
```

不入库（`.gitignore` 已排除，换机后需要自己准备）：

| 路径 | 内容 |
| --- | --- |
| `.venv/` | 本地 Python 环境 |
| `voice/` | 音色样本（含个人声音） |
| `.kaggle_jobs/` | 每次运行的 staging：请求 JSON、音色、拼接后的 main.py、metadata |
| `outputs/` | 下载回来的音频、result.json、kernel.log |

## 一次性准备（新机器）

```bash
cd /path/to/indextts
uv venv .venv
uv pip install --python .venv/bin/python "kaggle>=2.2.5" six
```

Kaggle 凭据二选一：

- 文件：`~/.kaggle/access_token`（只放 token 本身，一行）
- 环境变量：`export KAGGLE_API_TOKEN=...`

> 脚本会用**真实用户目录**（`pwd` 模块解析）去找 `~/.kaggle`，所以即使当前 shell 的 `$HOME` 被别的工具改写也能正常认证。

准备音色：建议 5–15 秒、干净、单人、无背景音乐。IndexTTS-2 只取参考音频前 15 秒，太长的文件先裁掉。

## 常用命令

合成一句话：

```bash
.venv/bin/python kaggle/run_kaggle_tts.py \
  --text "你好，这是通过 Kaggle 运行的 IndexTTS2。" \
  --voice voice/0328.wav
```

用 JSON 请求（可带情感参数）：

```bash
.venv/bin/python kaggle/run_kaggle_tts.py --request kaggle/examples/request.example.json
```

长文案放文件：

```bash
.venv/bin/python kaggle/run_kaggle_tts.py --text-file script.txt --voice voice/0226_ref.wav
```

指定 GPU / 强制 CPU：

```bash
# 批量 Kernel 抢 GPU 队列，T4 是 "NvidiaTeslaT4"（即 T4 x2）
.venv/bin/python kaggle/run_kaggle_tts.py --text "..." --voice voice/0328.wav --accelerator NvidiaTeslaT4
# 不想等 GPU 就用 CPU（能跑通但很慢）
.venv/bin/python kaggle/run_kaggle_tts.py --text "..." --voice voice/0328.wav --no-gpu
```

只生成 staging 不调用 Kaggle / 只推送不等待 / 续等已有任务：

```bash
.venv/bin/python kaggle/run_kaggle_tts.py --text "调试" --voice voice/0328.wav --dry-run
.venv/bin/python kaggle/run_kaggle_tts.py --text "后台跑" --voice voice/0328.wav --no-wait
.venv/bin/python kaggle/run_kaggle_tts.py --resume <job_id> --timeout 2400 --json-out outputs/last_run.json
```

音色已存在 Kaggle、模型已缓存成 Dataset，可省掉上传和 4.7 GB 下载：

```bash
.venv/bin/python kaggle/run_kaggle_tts.py \
  --text "..." --voice-dataset owner/my-voices --voice-file samples/me.wav \
  --model-dataset owner/indextts2-checkpoints
```

完整参数：`kaggle/run_kaggle_tts.py --help`。

## 请求 JSON 字段

命令行同名参数会覆盖 JSON。

| 字段 | 说明 |
| --- | --- |
| `text` | 必填，要合成的文稿 |
| `voice_path` | Kernel 内的音色路径；随请求上传时固定为 `/kaggle/input/indextts-request/voice.wav` |
| `emo_audio_path` | 情感参考音频（可选） |
| `emo_alpha` | 情感强度，默认 1.0 |
| `emo_vector` | 8 维向量 `[开心,生气,悲伤,害怕,厌恶,忧郁,惊讶,平静]` |
| `emo_text` / `use_emo_text` | 用文本描述情感 |
| `use_random` | 随机音色 |
| `max_text_tokens_per_segment` | 断句长度，默认 120 |
| `use_fp16` | 半精度，默认 true |
| `output_name` | 输出文件名，默认 `audio.wav` |
| `repo_ref` / `model_repo` | 固定 `v2.0.0` / `IndexTeam/IndexTTS-2` |
| `model_dir` | 已缓存的模型目录（挂载 Dataset 时用） |

## 产出

成功时 CLI 打印摘要，文件在 `outputs/<job_id>/`：

- `output/<job_id>/audio.wav` — 合成音频（22.05 kHz 单声道）
- `output/<job_id>/result.json` — `status`、`gpu`、`duration_sec`、`audio_bytes`、`elapsed_sec`
- `kernel.log`、`indextts-clone.log` — 远端运行日志，失败时先看这里

## 已知限制与踩过的坑

- **不是实时服务**：批量 Kernel 排 GPU 队列可能几十分钟；交互式 Notebook 能秒拿 T4，批量任务不一定。CLI 默认只等 3 小时（`--timeout`），超时后用 `--resume` 继续。
- **CPU 极慢**：T4 上 284 字约 6 分钟（含下载模型）；CPU 上同样内容会慢一个数量级，只建议验证链路。
- **冷启动成本**：每次都要 `uv sync` 并下载约 4.7 GB 权重。长期用请把 `checkpoints/` 做成 Kaggle Dataset，用 `--model-dataset` 挂载。
- **Dataset 挂载路径变了**：现在挂到 `/kaggle/input/datasets/<owner>/<slug>/`，旧的 `/kaggle/input/<slug>/` 不一定存在。`worker.py` 的 `resolve_input` 已做多路径兜底。
- **仓库自带占位模型**：index-tts 仓库里有一个只有 `config.yaml` 的 `checkpoints/`，会让"模型已存在"的判断误判、跳过下载，随后报 `qwen0.6bemo4-merge` 缺失。`bootstrap.py` 的 `model_ready()` 现在强制校验 `config.yaml`、`gpt.pth`、`s2mel.pth`、`bpe.model` 和 `qwen0.6bemo4-merge/` 齐全。
- **并发会互相覆盖**：请求 Dataset 用固定 slug，多个任务同时跑会踩，建议串行。
- **私密性**：Dataset 和 Kernel 都是私有的，音色和文稿不会公开。

## 给 AI 助手的入口

换机器后想快速接手，按这个顺序看：

1. `kaggle/run_kaggle_tts.py` 顶部常量与 `parse_args()` — 弄清输入和默认值
2. `kaggle/bootstrap.py` 的 `main()` — 远端执行步骤（clone → uv sync → 模型 → worker）
3. `kaggle/worker.py` 的 `main()` — 真正调用 `IndexTTS2.infer()`
4. 跑 `--dry-run` 生成 `.kaggle_jobs/<job_id>/`，对比 `kernel/main.py` 与 `kernel-metadata.json`
5. 端到端验证：短文案 + 15 秒音色，看到 `outputs/<job_id>/output/*/audio.wav` 即成功

排查顺序：先看 `outputs/<job_id>/kernel.log` 里的 `RESULT_JSON` 或 traceback，再看 Kaggle 运行页 `https://www.kaggle.com/code/<user>/indextts-clone`。
