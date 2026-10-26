# -*- coding: utf-8 -*-
"""
shot_detect.py —— 镜头边界检测（纯 numpy，无模型）

两条分支，对应真实素材里的两种切换，关键是"特征形状要匹配现象"：
  1) 硬切：画面突变 → 量化颜色直方图 + 帧间 L1 距离出现尖峰 → 自适应阈值 + NMS
  2) 渐变（叠化/淡入淡出）：画面连续变化、没有尖峰 → 直方图因量化会出现一串
     伪小尖峰，所以换成"平滑的 4x4 块均值"特征，用跨窗口累计差判定

这是"弱基线"。强基线是 TransNetV2、AutoShot（CVPRW 2023，面向短剧的 SBD）。
替换接口：把 detect() 换成模型推理即可，下游字段（boundaries/shots）保持不变。

输出：out/shots.json
"""

import glob
import json
import os

import numpy as np
from PIL import Image

from paths import FRAMES, OUT

BINS = 8           # 每通道量化级数 -> 8^3 = 512 维直方图
GRID = 4           # 块均值网格 4x4
WIN = 8            # 自适应阈值邻域半宽
K_SIGMA = 4.0      # 尖峰判定
D_MIN = 0.10       # 尖峰绝对下限
NMS = 4            # 非极大值抑制半径（帧）
GW = 6             # 渐变检测的窗口半宽
SG_THR = 0.30      # 渐变累计差阈值（块均值特征，0..1）
GAP_MERGE = 3      # 候选段之间小于该间隔就合并（避免一处叠化被切两半）


def load_frames(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"没找到帧：{pattern}，请先运行 make_footage.py")
    return paths


def features(path):
    """返回 (颜色直方图 512 维, 块均值 48 维)。"""
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    q = (arr.astype(np.uint8) >> 5).astype(np.int32)
    idx = q[..., 0] * 64 + q[..., 1] * 8 + q[..., 2]
    h = np.bincount(idx.ravel(), minlength=BINS ** 3).astype(np.float32)
    h /= max(h.sum(), 1e-6)

    hh, ww = arr.shape[0] // GRID, arr.shape[1] // GRID
    blocks = arr[:hh * GRID, :ww * GRID].reshape(GRID, hh, GRID, ww, 3).mean(axis=(1, 3))
    return h, (blocks.reshape(-1) / 255.0)


def l1(a, b):
    return float(np.abs(a - b).mean())


def detect(paths):
    feats = [features(p) for p in paths]
    H = np.stack([f[0] for f in feats])
    G = np.stack([f[1] for f in feats])
    n = len(H)

    # 帧间直方图距离（0..1）
    d = np.zeros(n, dtype=np.float32)
    d[1:] = 0.5 * np.abs(H[1:] - H[:-1]).sum(axis=1)

    # ---- 1) 硬切：尖峰 + 自适应阈值 ----
    spikes = []
    for i in range(1, n):
        lo, hi = max(1, i - WIN), min(n, i + WIN + 1)
        nb = np.concatenate([d[lo:i], d[i + 1:hi]])
        if nb.size < 3:
            continue
        thr = max(nb.mean() + K_SIGMA * nb.std(), D_MIN)
        if d[i] > thr:
            spikes.append(i)

    kept = []
    for c in sorted(spikes, key=lambda x: -d[x]):
        if all(abs(c - k) > NMS for k in kept):
            kept.append(c)
    hard = sorted(kept)

    # ---- 2) 渐变：平滑特征的跨窗口累计差 ----
    sg = np.zeros(n, dtype=np.float32)
    for i in range(n):
        a, b = i - GW, i + GW
        if a < 0 or b >= n:
            continue
        sg[i] = l1(G[b], G[a])

    near_cut = np.zeros(n, dtype=bool)
    for c in hard:
        near_cut[max(0, c - GW):min(n, c + GW + 1)] = True

    cand = (sg > SG_THR) & (~near_cut) & (d < 0.5)
    gradual, i = [], 0
    while i < n:
        if cand[i]:
            j = i
            while j + 1 < n and cand[j + 1]:
                j += 1
            if j - i + 1 >= 4:
                gradual.append((i, j))
            i = j + 1
        else:
            i += 1

    # 合并被极小空档切开的候选段
    merged = []
    for seg in gradual:
        if merged and seg[0] - merged[-1][1] <= GAP_MERGE:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(list(seg))

    boundaries = [{"frame": int(f), "type": "hard_cut", "score": round(float(d[f]), 4)}
                  for f in hard]
    for a, b in merged:
        c = (a + b) // 2
        boundaries.append({"frame": int(c), "type": "dissolve",
                           "score": round(float(sg[c]), 4), "band": [int(a), int(b) + 1]})
    boundaries.sort(key=lambda x: x["frame"])

    return {"n_frames": n, "boundaries": boundaries, "n_shots": len(boundaries) + 1}


def evaluate(pred, truth, tol=3):
    """按 ±tol 帧容差算 P/R/F1；渐变用"落在过渡带内"判定。"""
    def prf(gt, pd):
        used, tp = set(), 0
        for g in gt:
            for k, p in enumerate(pd):
                if k not in used and abs(g - p) <= tol:
                    tp += 1
                    used.add(k)
                    break
        return tp, len(gt), len(pd)

    pd_hard = [b["frame"] for b in pred["boundaries"] if b["type"] == "hard_cut"]
    pd_dis = [b["frame"] for b in pred["boundaries"] if b["type"] == "dissolve"]
    gt_hard = truth["hard_cuts"]
    gt_dis = [int((a + b) / 2) for a, b in truth.get("dissolves", [])]

    out = {}
    for name, gt, pd in (("hard_cut", gt_hard, pd_hard), ("dissolve", gt_dis, pd_dis)):
        tp, ng, npd = prf(gt, pd)
        p = tp / npd if npd else (1.0 if ng == 0 else 0.0)
        r = tp / ng if ng else 1.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        out[name] = {"tp": tp, "n_gt": ng, "n_pred": npd,
                     "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}

    out["n_shots_pred"] = pred["n_shots"]
    out["n_shots_gt"] = len(truth["shots"])
    return out


def main():
    paths = load_frames(os.path.join(OUT, "frames", "*.png"))
    res = detect(paths)
    with open(os.path.join(OUT, "truth.json"), encoding="utf-8") as fp:
        truth = json.load(fp)
    res["fps"] = truth["fps"]
    res["eval"] = evaluate(res, truth)
    with open(os.path.join(OUT, "shots.json"), "w", encoding="utf-8") as fp:
        json.dump(res, fp, ensure_ascii=False, indent=2)

    print("[shot_detect] 边界 %d 个 / 共 %d 帧 → 切出 %d 个镜头"
          % (len(res["boundaries"]), res["n_frames"], res["n_shots"]))
    for k, v in res["eval"].items():
        if isinstance(v, dict):
            print(f"  {k:9s} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} "
                  f"(tp={v['tp']}/{v['n_gt']}, pred={v['n_pred']})")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
