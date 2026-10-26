# -*- coding: utf-8 -*-
"""
check_env.py —— 环境自检：Python / 依赖包 / ffmpeg / GPU

只用标准库，所以在一个"什么都没装"的机器上也能跑，不会崩。

    python check_env.py

退出码：0 = 可以跑 demo；1 = 缺必需项
"""

import importlib
import os
import platform
import shutil
import subprocess
import sys

OK, NO, WARN, INFO = "[OK]  ", "[MISS]", "[WARN]", "[INFO]"


def ver(mod_name):
    try:
        m = importlib.import_module(mod_name)
        return getattr(m, "__version__", "已安装")
    except Exception:
        return None


def has_cmd(name):
    return shutil.which(name) is not None


def run(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        return f"<执行失败: {e}>"


def main():
    lines, blockers, warns = [], [], []

    def add(status, item, detail):
        lines.append((status, item, detail))

    # ---------- 1. Python ----------
    pv = sys.version.split()[0]
    ok_py = sys.version_info >= (3, 9)
    add(OK if ok_py else NO, "Python", f"{pv}  ({platform.architecture()[0]})")
    if not ok_py:
        blockers.append("Python 版本低于 3.9，装 3.11 或 3.12")
    add(INFO, "解释器路径", sys.executable)
    add(INFO, "操作系统", f"{platform.system()} {platform.release()}")

    # ---------- 2. 必需包 ----------
    for mod, label, why in [
        ("numpy", "numpy", "矩阵运算，demo 核心依赖"),
        ("PIL", "Pillow", "读写图片帧，demo 核心依赖"),
    ]:
        v = ver(mod)
        add(OK if v else NO, label, v or f"缺失 —— {why}")
        if not v:
            blockers.append(f"pip install {label.lower() if label != 'Pillow' else 'pillow'}")

    # ---------- 3. 可选包（L2 跨模态）----------
    l2_missing = []
    for mod, label, why in [
        ("torch", "torch", "接 CLIP / InternVideo2 做跨模态检索（L2 步骤才需要）"),
        ("transformers", "transformers", "加载 Chinese-CLIP 权重"),
        ("cv2", "opencv-python", "可选：更快的抽帧/缩放"),
        ("faiss", "faiss-cpu", "可选：百万级向量索引"),
    ]:
        v = ver(mod)
        add(OK if v else WARN, label, v or f"未装 —— {why}")
        if not v and mod in ("torch", "transformers"):
            l2_missing.append(label)
    if l2_missing:
        warns.append(f"L2 需要：pip install torch transformers  "
                     f"(缺 {'/'.join(l2_missing)}，当前 demo 会自动跳过 B2)")

    # ---------- 4. ffmpeg ----------
    if has_cmd("ffmpeg"):
        out = run(["ffmpeg", "-version"]).splitlines()
        add(OK, "ffmpeg", out[0] if out else "已安装")
    else:
        add(WARN, "ffmpeg", "未装 —— 换真视频抽帧要用：winget install -e --id Gyan.FFmpeg")
        warns.append("ffmpeg 缺失：L1 换成真视频时会卡在抽帧；合成素材的 demo 不受影响")

    # ---------- 5. GPU ----------
    if has_cmd("nvidia-smi"):
        out = run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
        add(OK, "NVIDIA GPU", out.strip().splitlines()[0] if out.strip() else "检测到显卡")
        try:
            import torch
            add(OK if torch.cuda.is_available() else WARN, "torch.cuda",
                "可用" if torch.cuda.is_available() else "torch 装了但看不到 CUDA")
        except Exception:
            add(INFO, "torch.cuda", "torch 未装，无法判断")
    else:
        add(WARN, "NVIDIA GPU", "未检测到 nvidia-smi —— 本机大概是核显/无独显")
        warns.append("无 NVIDIA 显卡：L2（接 CLIP）请走 Kaggle Notebook 或 Colab 的免费 T4，"
                     "不要在本地硬跑")

    # ---------- 6. 虚环境检查 ----------
    if sys.prefix == sys.base_prefix:
        add(WARN, "虚拟环境", "当前是系统 Python（建议用 venv，避免以后 torch 与系统包打架）")
    else:
        add(OK, "虚拟环境", sys.prefix)

    # ---------- 输出 ----------
    w1 = max(len(x[1]) for x in lines) + 2
    w2 = max(len(x[2]) for x in lines) + 2
    print("=" * (8 + w1 + w2))
    print(f"{'状态':<8}{'项目':<{w1}}{'详情'}")
    print("-" * (8 + w1 + w2))
    for s, a, b in lines:
        print(f"{s:<8}{a:<{w1}}{b}")
    print("=" * (8 + w1 + w2))

    if warns:
        print("\n提示：")
        for w in warns:
            print("  - " + w)

    print()
    if blockers:
        print(f"{NO} 还不能跑 demo，先解决：")
        for i, b in enumerate(blockers, 1):
            print(f"  {i}. {b}")
        return 1

    print(f"{OK} 可以跑 demo 了：")
    print("     cd demo && python run_demo.py")
    if any("torch" in w for w in warns):
        print(f"{INFO} 接跨模态模型（L2）还差 torch + transformers，不影响当前 demo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
