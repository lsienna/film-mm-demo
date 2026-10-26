# -*- coding: utf-8 -*-
"""
make_footage.py —— 生成一段"影视素材"合成测试片（无需 ffmpeg / 无需联网 / 无需 GPU）

为什么要合成素材？
    影视素材有版权，本地也不可能放几十 GB 的片子。这个脚本用 Pillow+numpy
    造一段 20 秒、7 个镜头、含 1 处叠化、含画外音的"伪影视片段"，目的是把
    【镜头切分 → 关键帧 → 台词 → 人物 → 结构化 JSON → 检索评测】整条流水线
    先跑通。换成真片时，只需要改 shot_detect.py 的输入（见 README）。

产物（写到 out/）：
    frames/000000.png ...      逐帧图像（代替解码后的视频帧）
    truth.json                 人工标注真值：镜头边界、叠化区间、台词、人物在场
    asr.json                   模拟 ASR 输出（含 2 处识别错误，用于后面算对齐准确率）
"""

import json
import os

import numpy as np
from PIL import Image

from paths import FRAMES, OUT

W, H, FPS = 320, 180, 10

# 人物原型：真实系统里这里是"人脸 embedding 聚类得到的角色原型"，
# 这里用固定色块代替，方便用纯 numpy 做"人物在场检测"。
CHARACTERS = {
    "C1": {"name": "林晓", "color": [205, 62, 62]},
    "C2": {"name": "周航", "color": [62, 62, 205]},
}

# (shot_id, 帧数, 场景标签, 底色, 在场人物, 台词说话人, 台词)
SHOTS = [
    ("S001", 26, "办公室-白天", [232, 226, 212], ["C1"], "C1", "这个版本的调色我觉得还是太亮了"),
    ("S002", 22, "特写-林", [198, 124, 112], ["C1"], "C1", "再压一档，把高光收回来"),
    ("S003", 18, "走廊-夜", [34, 42, 72], ["C2"], "C2", "导演那边催了，今晚必须交片"),
    ("S004", 24, "街道-白天", [152, 182, 218], [], "C1", "我先把第九场重剪一遍"),
    ("S005", 20, "回忆-雨", [88, 108, 130], ["C1", "C2"], "C2", "那年也是这样的雨"),
    ("S006", 26, "办公室-白天", [230, 224, 210], ["C1"], "C1", "行了，按这版走"),
    ("S007", 22, "天台-夜", [22, 26, 46], [], "C1", "收工"),
]

DISSOLVE_LEN = 12  # S003 与 S004 之间用叠化过渡（考验"渐变镜头切换"检测）


def render_shot(shot_idx, local_t, n_local, rng):
    """渲染某个镜头在自身时间轴 local_t 处的一帧。"""
    _, _, _, base, present, _, _ = SHOTS[shot_idx]
    img = np.zeros((H, W, 3), dtype=np.float32)
    img[:] = np.array(base, dtype=np.float32)

    # 轻微纵向渐变，模拟光照
    grad = np.linspace(-14, 14, H, dtype=np.float32)[:, None, None]
    img = img + grad

    # 运动光斑，模拟镜头内运动
    cx = int(30 + (W - 60) * (0.5 + 0.5 * np.sin(2 * np.pi * (local_t / max(n_local, 1)))))
    img[:, max(0, cx - 18):max(0, cx + 18), :] += 26.0

    # 人物：站在画面里，随时间横向走动
    for k, cid in enumerate(present):
        color = np.array(CHARACTERS[cid]["color"], dtype=np.float32)
        x0 = 24 + k * 120 + int(10 * np.sin(2 * np.pi * local_t / max(n_local, 1) + k))
        x1 = min(W, x0 + 34)
        y0 = H - 96
        box = img[y0:H, x0:x1, :]
        if box.size:
            box[:] = color

    img += rng.normal(0, 5.0, img.shape)
    return np.clip(img, 0, 255)


def main():
    os.makedirs(os.path.join(OUT, "frames"), exist_ok=True)
    rng = np.random.default_rng(20240521)

    # 1) 先排好每个镜头的帧区间（暂按硬切）
    starts, cursor = [], 0
    for sid, n, *_ in SHOTS:
        starts.append(cursor)
        cursor += n
    n_frames = cursor

    # 2) 把 S003|S004 这一处硬切改成叠化
    bidx = 3                      # 边界：S003 结束 / S004 开始
    b = starts[bidx]
    half = DISSOLVE_LEN // 2
    dis_start, dis_end = b - half, b - half + DISSOLVE_LEN

    frames = []
    for i in range(n_frames):
        if dis_start <= i < dis_end:
            a = (i - dis_start + 1) / (DISSOLVE_LEN + 1)      # 0→1 线性混合
            f_left = render_shot(bidx - 1, i - starts[bidx - 1], SHOTS[bidx - 1][1], rng)
            f_right = render_shot(bidx, i - starts[bidx], SHOTS[bidx][1], rng)
            f = (1 - a) * f_left + a * f_right
        else:
            k = max(j for j in range(len(SHOTS)) if starts[j] <= i)
            f = render_shot(k, i - starts[k], SHOTS[k][1], rng)
        frames.append(np.clip(f, 0, 255).astype(np.uint8))

    # 3) 落盘
    for i, f in enumerate(frames):
        Image.fromarray(f).save(os.path.join(OUT, "frames", f"{i:06d}.png"))

    # 4) 真值：镜头边界（叠化区间记为渐变边界）
    shots_gt, cuts_gt = [], []
    for k, (sid, n, label, _, present, speaker, line) in enumerate(SHOTS):
        t_in, t_out = starts[k], starts[k] + n
        if k == bidx:
            t_in = dis_end                     # 叠化结束后才算新镜头
        if k == bidx - 1:
            t_out = dis_start
        shots_gt.append({
            "shot_id": sid, "in": t_in, "out": t_out, "n_frames": t_out - t_in,
            "scene_label": label, "characters_present": present,
            "dialogue": {"speaker": speaker, "text": line},
        })
        if k > 0 and k != bidx:
            cuts_gt.append(starts[k])

    truth = {
        "fps": FPS, "width": W, "height": H, "n_frames": n_frames,
        "characters": CHARACTERS,
        "shots": shots_gt,
        "hard_cuts": cuts_gt,
        "dissolves": [[dis_start, dis_end]],
    }
    with open(os.path.join(OUT, "truth.json"), "w", encoding="utf-8") as fp:
        json.dump(truth, fp, ensure_ascii=False, indent=2)

    # 5) 模拟 ASR + 说话人日志（真实系统：Whisper + pyannote；这里故意埋 2 处错字）
    typos = {"调色": "调色", "高光": "高先", "重剪": "重简"}
    asr = []
    for s in shots_gt:
        text = s["dialogue"]["text"]
        for a, b2 in typos.items():
            text = text.replace(a, b2)
        asr.append({
            "shot_id": s["shot_id"],
            "t_in": round(s["in"] / FPS, 2),
            "t_out": round(s["out"] / FPS, 2),
            "speaker": s["dialogue"]["speaker"],
            "text": text,
            "wer_hint": "模拟输出：含替换错误，用于演示对齐准确率的算法口径",
        })
    with open(os.path.join(OUT, "asr.json"), "w", encoding="utf-8") as fp:
        json.dump(asr, fp, ensure_ascii=False, indent=2)

    print(f"[footage] frames={n_frames}  shots={len(shots_gt)}  "
          f"hard_cuts={len(cuts_gt)}  dissolve={DISSOLVE_LEN}帧 -> {OUT}")


if __name__ == "__main__":
    main()
