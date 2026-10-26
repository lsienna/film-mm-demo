# -*- coding: utf-8 -*-
"""
paths.py —— 统一的路径解析

为什么单独抽出来：Kaggle / Colab 里的代码目录（/kaggle/input/... 或 /content/...）
往往是只读的，而 demo 需要往 out/ 写帧和报告。所以输出目录必须可重定向：

    # 本地：默认写到 demo/out/
    python run_demo.py

    # Kaggle / Colab / 只读代码目录：
    export DEMO_OUT=/kaggle/working/out      # Linux
    $env:DEMO_OUT="D:\\tmp\\demo_out"        # Windows PowerShell
    python run_demo.py
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("DEMO_OUT") or os.path.join(HERE, "out")
OUT = os.path.abspath(OUT)
FRAMES = os.path.join(OUT, "frames")
KEYFRAMES = os.path.join(OUT, "keyframes")
