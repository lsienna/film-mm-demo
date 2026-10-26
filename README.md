# demo —— 影视素材跨模态语义理解 · 最小闭环

纯 `numpy` + `Pillow` 实现，**不需要 GPU、不需要联网、不需要装 torch**。
目的不是刷分，而是把「镜头级结构化 → 跨模态召回 → 重排 → 指标口径」
这条链的**数据契约和评测算法**先钉死。

```bash
cd demo
python run_demo.py          # 一键跑通，产出写到 out/
```

Python 3.9+ 即可，L0/L1/L3 **只需要 numpy 和 Pillow**。

---

## 0. 装环境（Windows，一次搞定）

### 0.1 先自检，看缺什么

```powershell
python check_env.py
```

只用标准库，什么都没装也能跑。输出会明确告诉你缺哪项、该装什么。

### 0.2 装 Python（如果 `python` 命令不存在）

```powershell
# 方式 A（已有 Anaconda，最省事）：直接用 conda，见 0.3-B
# 方式 B：winget（Win11 自带）
winget install -e --id Python.Python.3.12

# 方式 C：官网下载 https://www.python.org/downloads/
#         安装时务必勾选 "Add python.exe to PATH"
```

装完**关掉 PowerShell 重新开一个**，验证：

```powershell
python --version      # 应输出 Python 3.12.x
```

### 0.3 建虚拟环境（二选一）

**方式 A · 已装 Anaconda（推荐，省掉装解释器）**

```
打开「Anaconda Prompt」（不是普通 PowerShell）
conda create -n filmdemo python=3.11 -y
conda activate filmdemo
cd /d "C:\Users\ls\Documents\...\影视跨模态_两周启动\demo"
pip install -r requirements.txt
python check_env.py
```

本地只需要 numpy + Pillow（Anaconda base 里其实就有）。**torch 和 ffmpeg 不用在本地装** ——
torch 在 Kaggle 上跑，ffmpeg 只在你要处理真视频抽帧时才需要。

> `conda activate` 在 PowerShell 里常报 "Run 'conda init' before 'conda activate'"。
> 两个解法：① 用 Anaconda Prompt；② 在 PowerShell 跑一次 `conda init powershell` 再重开终端。

**方式 B · 没有 Anaconda，用 venv**

```powershell
cd 影视跨模态_两周启动\demo
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # 命令行前面应出现 (.venv)

# 如果报"禁止运行脚本"，先执行一次：
# Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 0.4 装依赖

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt          # numpy + Pillow，几秒钟
```

国内网络慢就加镜像：

```powershell
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 0.5 装 ffmpeg（换真视频抽帧才需要，合成素材的 demo 不用）

```powershell
winget install -e --id Gyan.FFmpeg
# 或手动下载 https://www.gyan.dev/ffmpeg/builds/ 解压后把 bin 目录加进 PATH
```

验证：`ffmpeg -version`

### 0.6 【L2 才需要】torch + transformers

**先确认有没有 NVIDIA 显卡**：`nvidia-smi`。有输出才装 GPU 版，否则装 CPU 版。

```powershell
# 有 N 卡
pip install -r requirements-gpu.txt

# 无 N 卡（只有核显）——本地跑 CLIP 会很慢，建议改用 Kaggle/Colab
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers ftfy
```

模型权重从 HuggingFace 拉不动的话，先设镜像再跑：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

### 0.7 验证

```powershell
python check_env.py       # 全 [OK] 或只剩 [WARN] 就行
python run_demo.py        # 能出报告 = 环境通了
```

---

## 0.8 在 Kaggle / Colab 上跑

**能跑，而且 L2（接 CLIP）建议就在云端跑**——本机没有 N 卡。

> **Kaggle 镜像自带 numpy / Pillow / torch / transformers / opencv**（版本随镜像更新）。
> 所以第一件事永远是 `python check_env.py`，确认缺什么再装。绝大多数情况**一个包都不用 pip**，
> 省时间还避免版本冲突。

### 一个必须先知道的坑

Kaggle 的 `/kaggle/input`（数据集挂载点）和 Colab 挂载的 Drive **都是只读的**。
代码放在那里，`run_demo.py` 写 `out/` 会直接 `PermissionError`。

所以输出目录做成了可重定向，用环境变量 `DEMO_OUT` 指定：

```bash
export DEMO_OUT=/kaggle/working/out     # Linux（Kaggle/Colab）
```

不设就默认写 `demo/out/`（本地行为不变）。

### 把代码弄上去的三种方式

| 方式 | 适合 | 做法 |
|---|---|---|
| **GitHub**（推荐） | 两周要反复改，还要当申请书附件 | 推到仓库，notebook 里 `!git clone` |
| **Kaggle Dataset** | 一次性上传整个文件夹 | 打包 zip 传成 Dataset，`copytree` 到 working |
| **Colab + Drive** | 已有 Google Drive | 挂载 Drive，`copytree` 到 `/content` |

### Kaggle 完整流程（可直接粘）

```python
# ---------- cell 1：准备 ----------
import os, shutil
os.environ["DEMO_OUT"] = "/kaggle/working/out"        # 关键：写到可写目录
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"   # 权重镜像

# 二选一
SRC = "/kaggle/input/你的数据集名/demo"                 # A. 用 Dataset
# B. 用 GitHub 就直接 clone 到 WORK

WORK = "/kaggle/working/demo"
if not os.path.exists(WORK):
    shutil.copytree(SRC, WORK)          # 若已 git clone 则跳过这行
os.chdir(WORK)
print(sorted(os.listdir(".")))
```

```python
# ---------- cell 2：跑（不需要 GPU，CPU 也行）----------
!python run_demo.py
```

```python
# ---------- cell 3：L2 接跨模态模型（要开 GPU）----------
# 先看 Kaggle 镜像自带了什么，别无脑 pip —— 自带 torch/transformers 的话这步可跳过
!python check_env.py
# 只有报 [MISS] 才装：
# !pip -q install torch transformers ftfy
!python run_demo.py          # 会自动多出 B2 那一行
```

```python
# ---------- cell 4：把结果存成可下载的产物 ----------
import shutil
shutil.make_archive("/kaggle/working/results", "zip", "/kaggle/working/out")
print(open("/kaggle/working/out/report.md", encoding="utf-8").read())
```

**Kaggle 设置项**：右侧 Settings → `Internet: On`（需手机验证账号）→ `Accelerator: GPU T4 x2`。
跑完点 **Save Version**，`/kaggle/working` 里的东西才会存下来。

### Colab 版差异

```python
from google.colab import drive
drive.mount("/content/drive")
import os, shutil
os.environ["DEMO_OUT"] = "/content/out"
SRC = "/content/drive/MyDrive/影视跨模态/demo"
shutil.copytree(SRC, "/content/demo", dirs_exist_ok=True)
os.chdir("/content/demo")
!pip -q install -r requirements.txt
!python run_demo.py
```

### 两边怎么选

| | Kaggle | Colab（免费） |
|---|---|---|
| GPU | T4 x2 / P100，约 30 h/周 | T4，不保证 |
| 单次时长 | 12 h | 12 h，但常被随机断开 |
| 持久化 | Save Version 存 `/kaggle/working` | 必须挂 Google Drive |
| 联网 | 需手动开 + 手机验证 | 默认可用 |
| 适合 | **本项目**（长会话、配额稳） | 临时试一下 |

结论：**用 Kaggle**。本地只做 L0/L1/L3（不需要 GPU），L2 全部丢到 Kaggle。

---

## 1. 它做了什么事

```
make_footage.py    合成一段 15.8 秒的"伪影视素材"：7 个镜头、5 处硬切、1 处 12 帧叠化、
                   2 个角色、7 句台词（含 2 处故意埋的 ASR 错字）、1 处画外音
      ↓
shot_detect.py     镜头边界检测：硬切用颜色直方图尖峰，叠化用平滑块均值的跨窗累计差
      ↓
structure.py       镜头级结构化：关键帧、时间码、视觉属性、人物在场、场景聚合、台词、事件
                   → out/structure.json（数据契约） + out/entities.json + out/relations.csv
      ↓
retrieval.py       三路检索：BM25 文本召回 / 结构化硬约束 / 跨模态 CLIP（本地无权重则跳过）
      ↓
evaluate.py        Hit@1、Hit@5、MRR、mAP + 对齐准确率（两种口径）+ 自测用例
      ↓
out/report.md      自动生成的指标报告，可直接截图进申请书
```

## 2. 跑出来是什么数（本地无 GPU 时）

| 检索器 | Hit@1 | Hit@5 | MRR |
|---|---|---|---|
| B1 纯文本召回（BM25） | 69.2% | 92.3% | 0.783 |
| B1s 结构化约束 + BM25 | 76.9% | 92.3% | 0.821 |
| B2 跨模态 CLIP | SKIP（无权重） | — | — |
| B3 融合（无 B2 时 = B1s） | 76.9% | 92.3% | 0.821 |

台词—人物绑定准确率：**57.1%（4/7）**；拆开看是「单人镜头 4/4、多人同框 0/1、画外音 0/2」。

> ⚠️ **13 条查询、7 个镜头，样本量太小，这些百分比只用于验证口径，不能当性能结论。**
> 真实数值要等你换成真素材 + 扩到 100 条查询 + 接上 CLIP 才有意义。

## 3. 哪些是真的，哪些是占位

| 模块 | 状态 | 说明 |
|---|---|---|
| 硬切检测 | **真实算法** | 量化直方图 + 自适应阈值 + NMS，可处理任意真视频 |
| 叠化检测 | **真实算法** | 平滑块均值 + 跨窗累计差；换成真素材时阈值需重调 |
| 关键帧 / 时间码 | 真实 | 中间帧 + `HH:MM:SS:FF` |
| BM25 检索 | 真实 | 中文字 bigram 分词 |
| 查询理解 + 结构化过滤 | 真实（规则式） | 抽取人物名/人数/亮度/色温/饱和度约束 |
| 指标计算 | 真实 | 含自测，可直接用于申请书写口径 |
| 人物在场 | **占位** | 用固定色块当"人脸聚类原型"，真实系统换人脸检测+聚类 |
| 场景聚合 | **占位** | 只做相邻镜头主色聚合，本片 7 个镜头一个都没合并 —— 说明光靠主色不够，真实系统要用 Rao 等 (CVPR 2020) 的场景分割 |
| 事件抽取 | **占位** | 关键词词表；真实系统用事件抽取/剧本对齐 |
| 台词 | **模拟** | `asr.json` 是脚本造的，并故意埋了 2 处错字；真实系统换 Whisper |
| 说话人日志 | **模拟** | 真实系统换 pyannote |
| 跨模态向量 | **空插槽** | `structure.json` 里 `embedding: null`，`retrieval.CLIPEncoder` 留好了接口 |

## 4. 怎么换成真素材

### 4.1 抽帧（不需要 ffmpeg 也能做，但建议装）

```bash
# 装 ffmpeg 后
ffmpeg -i my_clip.mp4 -vf fps=10 out/frames/%06d.png
```

然后把 `make_footage.py` 那一步跳过，直接跑 `shot_detect.py`。
**注意**：`shot_detect.py` 的阈值是按合成素材调的（画面干净、噪声小），真素材需要重调：

- `D_MIN`（硬切下限）：真素材有手持抖动/闪光，通常要调高到 0.2~0.3
- `K_SIGMA`：真素材噪声大，通常 3.0~5.0
- `SG_THR`（叠化阈值）：跟镜头运动强度强相关，动得厉害就要调高

调参方法：在 20~30 个镜头的真素材上标一遍边界，网格搜 3 个参数，画 P/R/F1 表。
**这张表本身就是申请书里"可行性"的证据。**

### 4.2 接真模型（L2，需要 GPU）

只改 `retrieval.py` 的 `CLIPEncoder`，**别改函数签名**，其它代码一行不用动：

```python
# 中文首选（影视台词是中文）
CLIPEncoder("OFA-Sys/chinese-clip-vit-large-patch16")

# 或者用视频模型（需要额外处理帧序列，工作量大一档）
# InternVideo2 / UMT：https://github.com/TencentARC/UMT
```

依赖：`pip install torch transformers ftfy pillow`。显存：base 模型约 4~6 GB。

接上之后 `run_demo.py` 会自动多出一行 B2，并给出 B1/B1s/B2/B3 四行对比表。

### 4.3 其它模块替换顺序（建议按这个顺序，否则不知道提升来自谁）

1. 先接 CLIP（B2）—— 拿到跨模态基线
2. 再换 TransNetV2 切分 —— 影响所有下游指标
3. 再换 Whisper —— 台词质量决定 T1 类查询
4. 最后换 pyannote —— 只影响对齐指标，不影响检索

## 5. 文件说明

| 文件 | 作用 |
|---|---|
| `make_footage.py` | 合成测试片 + 真值 + 模拟 ASR |
| `shot_detect.py` | 镜头边界检测（硬切 + 渐变）+ P/R/F1 |
| `structure.py` | 镜头级结构化 + 实体表 + 关系表 |
| `retrieval.py` | BM25 / 查询理解 / 结构化过滤 / CLIP 插槽 / 融合 |
| `evaluate.py` | Hit@1 等检索指标 + 对齐准确率（两种口径）+ 自测 |
| `run_demo.py` | 串联全流程 + 生成报告 |
| `out/report.md` | 指标报告（自动生成） |
| `out/structure.json` | 结构化输出示例 —— **这个 schema 就是申请书里的"数据契约"** |
| `out/relations.csv` | 三元组，可直接导入 Neo4j / NebulaGraph |

## 6. 你该在这个 demo 上改什么（对应两周计划）

- **D2**：给 `evaluate.py` 加 R@10，验证自测还能过
- **D5**：把 `shot_detect.py` 的阈值改成命令行参数，产出一张 P/R/F1 随阈值变化表
- **D6**：换成你手机录的 2 分钟真视频，记录"哪里崩了"
- **D9**：加第二路召回（颜色直方图 / 构图特征），做单路 vs 融合的消融表
- **D11**：把查询集扩到 100 条、镜头扩到 30 个，重算所有指标
- **D12**：用 `out/report.md` 的数写"指标可达性论证"
