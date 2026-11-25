# Quoridor Phase 1 训练断点 · 2026-11-25

课程 `phase1-course-20261125-restart-original` 暂停于 Level 2。MLP 已按原门槛连续三次通过，CNN32 等待续训。两组均使用本机生成的训练对局。

| 模型 | 已保存的 PPO 更新 | 已结算训练局数 | 保存时对 Level 2 搜索的正式评估 | 下一步 |
|---|---:|---:|---|---|
| MLP · CPU | 662 | 64,047 | 第 652、657、662 次更新分别赢 140、144、146 局，各 200 局，均为零截断、零非法 | 保留合格模型 |
| CNN32 · Apple MPS | 2,832 | 204,065 | 105 胜、57 负、38 截断，零非法，连续达标 0 次 | 从 `cnn32-seed0/latest.pt` 进行第 2,833 次更新，下一次评估在累计至少 204,500 局时触发 |

CNN32 的 `best.pt` 来自第 2,290 次更新（116 胜、50 截断，200 局）。续训使用 `latest.pt`，包含参数、Adam 状态、训练配置、采样中的棋局和评估进度。MLP 的 `latest.pt`、`best.pt`、`qualified.pt` 内容相同。

Level 2 训练池为 80% 中深度搜索、20% Random。晋级评估对中深度搜索每次 200 局，先后手各 100 局，每次至少 140 胜、截断少于 10 局、非法动作 0 次，连续通过三个冻结开局组。两个模型均达标后，课程调度才可进入 Level 3。Level 1 沿用之前的资格记录，其中 MLP 对 Random 采用当时批准的调整门槛。归档的旧 Level 2 结果独立于本次课程。

## 训练面板

课程清单中 Level 1 已完成，Level 2 已暂停，Level 3–5 待开始。点选模型可查看更新编号、正式评估和训练指标。以下为暂停后在本机面板核对的画面。

[MLP 达标面板](evidence/phase1-level2-mlp-20261125.png) · [CNN32 暂停面板](evidence/phase1-level2-cnn32-20261125.png)

## 下载与校验

使用一个全新的目录恢复。以下命令需安装 Git、GitHub CLI、zstd；tar 需支持 zstd。恢复包采用仓库相对路径，解包会写入同名文件。

```sh
git clone --branch phase1-paused-20261125 --depth 1 https://github.com/hy1195571214-byte/quoridor-lab.git
cd quoridor-lab
mkdir checkpoint-download
gh release download phase1-paused-20261125 -R hy1195571214-byte/quoridor-lab \
  -D checkpoint-download -p resume.tar.zst -p SHA256SUMS
(cd checkpoint-download && awk '$2 == "resume.tar.zst"' SHA256SUMS | shasum -a 256 -c -)
tar -xf checkpoint-download/resume.tar.zst
```

`resume.tar.zst` 包含当前模型、恢复所需的 Level 1 来源、冻结 Arena、监控状态和指标。完整历史还包括保留的 Phase 1 搜索试验、Level 1 原训练与专项/对照组、旧 Level 2、重训后的所有中间检查点、日志、评估棋谱和随机墙专项实验。分卷可以单独解压。`CONTENTS.json` 列出每卷中的原始路径。本机进程 ID 和锁文件由启动程序重新生成。

需要完整回溯时，从仓库根目录执行：

```sh
gh release download phase1-paused-20261125 -R hy1195571214-byte/quoridor-lab \
  -D checkpoint-download -p 'history-*.tar.zst' -p CONTENTS.json
(cd checkpoint-download && shasum -a 256 -c SHA256SUMS)
for archive in checkpoint-download/history-*.tar.zst; do tar -xf "$archive"; done
```

## 恢复 CNN32

本次记录环境为 Python 3.11.15、PyTorch 2.13.0、Apple Silicon MPS。先安装依赖并确认 MPS 可用。已配置好的本机训练环境可直接使用原虚拟环境。

```sh
cd training
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[learning]' 'torch==2.13.0'
.venv/bin/python -c 'import torch; assert torch.backends.mps.is_available(), "需要可用的 Apple MPS"'
```

下载和解包后训练仍处于暂停状态。准备好续训时，在 `training` 目录执行以下命令，只启动尚未合格的 CNN32：

```sh
run=data/training-runs/phase1-course-20261125-restart-original
.venv/bin/python -u -m quoridor_lab.phase1_curriculum --root "${run}-level2" --architecture cnn32
```

如要恢复自动课程调度，在 CNN32 已开始采样后，从另一终端的 `training` 目录启动：

```sh
.venv/bin/python -u -m quoridor_lab.phase1_course \
  --root data/training-runs/phase1-course-20261125-restart-original \
  --level1 data/training-runs/phase1-level1-random-special-20261125 \
  --pilot data/training-runs/phase1-search-20261124 --original-gates-only
```

## 核心模型 SHA-256

| Level 2 文件 | SHA-256 |
|---|---|
| `mlp-seed0/qualified.pt` | `6cb67ca9c60944cdcba2eaa1755afcb24b68f9b62a57bf9e505a37b1d96e3d5a` |
| `cnn32-seed0/latest.pt` | `080dce305a52bb9b80cef685232fadc17d64e8a3c4cee2643a5dfb5c94739887` |
| `cnn32-seed0/best.pt` | `86268978593c8ab26f3158701d808478d87d77da88a783fd07b6a27ccce7f8a1` |
