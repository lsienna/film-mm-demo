# -*- coding: utf-8 -*-
"""
structure.py —— 镜头级结构化 + 生产要素关联（指南里的"镜头级结构化"和"知识化组织"）

输入：out/shots.json（边界）、out/asr.json（台词+说话人日志）、out/frames/*
输出：
    out/keyframes/S00X.png        每个镜头的关键帧
    out/structure.json            镜头级结构化记录（含生产要素引用）
    out/entities.json             实体表：人物/场景/事件/台词/版本/素材
    out/relations.csv             关系表（三元组），可直接灌图数据库

注意：这里的"人物识别"用固定色块当人物原型，"场景"用主色聚类，"事件"用词表规则。
都是占位弱基线——真实系统换成 人脸聚类 / 场景分类 / 事件抽取模型，
**但对外输出的 JSON schema 不要变**，因为申请书里的"数据契约"就是这一层。
"""

import csv
import json
import os
import shutil

import numpy as np
from PIL import Image

from paths import OUT

PERSON_PIXEL_THR = 0.012     # 人物色块占画面比例阈值
SCENE_COLOR_THR = 0.12       # 相邻镜头主色距离阈值（小于则归为同一场景）

# 事件词表（占位规则）。真实系统用事件抽取/剧本对齐。
EVENT_LEXICON = {
    "调色": "后期调色", "高光": "后期调色", "压一档": "后期调色",
    "催": "催促", "交片": "催促",
    "重剪": "剪辑", "版本": "剪辑",
    "雨": "回忆-雨天",
    "收工": "收工",
}


def tc(frame, fps):
    s, f = divmod(int(frame), fps)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"


def to_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def visual_attrs(arr):
    """弱视觉属性抽取：亮度/色温/饱和度/运动量/人物在场。"""
    luma = float((0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).mean())
    r, g, b = [float(arr[..., i].mean()) for i in range(3)]
    sat = float((arr.max(axis=2) - arr.min(axis=2)).mean())
    warmth = r - b
    luma_tag = "白天" if luma > 150 else ("黄昏" if luma > 90 else ("昏暗" if luma > 70 else "夜晚"))
    temp_tag = "暖色调" if warmth > 12 else ("冷色调" if warmth < -12 else "中性色")
    sat_tag = "高饱和" if sat > 70 else ("中饱和" if sat > 30 else "低饱和")
    return {"luma": round(luma, 1), "warmth": round(warmth, 1), "sat": round(sat, 1),
            "luma_tag": luma_tag, "temp_tag": temp_tag, "sat_tag": sat_tag}


def persons_in(arr, characters):
    """按参考色占比判断人物是否在场（占位实现，真实系统=人脸检测+聚类）。"""
    hit = []
    for cid, info in characters.items():
        ref = np.array(info["color"], dtype=np.float32)
        m = (np.abs(arr - ref).max(axis=2) < 30)
        if m.mean() >= PERSON_PIXEL_THR:
            hit.append({"char_id": cid, "name": info["name"], "score": round(float(m.mean()), 4)})
    hit.sort(key=lambda x: -x["score"])
    return hit


def main():
    with open(os.path.join(OUT, "shots.json"), encoding="utf-8") as fp:
        shots = json.load(fp)
    with open(os.path.join(OUT, "asr.json"), encoding="utf-8") as fp:
        asr = json.load(fp)
    with open(os.path.join(OUT, "truth.json"), encoding="utf-8") as fp:
        truth = json.load(fp)

    fps = truth["fps"]
    characters = truth["characters"]           # 剧中角色表（生产系统里来自选角/定妆资料）
    os.makedirs(os.path.join(OUT, "keyframes"), exist_ok=True)

    # 1) 边界 → 镜头区间
    bounds = [b["frame"] for b in shots["boundaries"]]
    edges = [0] + bounds + [shots["n_frames"]]
    spans = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]

    records, prev_blocks = [], None
    scene_seq = 0
    for k, (fin, fout) in enumerate(spans):
        sid = f"S{k + 1:03d}"
        mid = (fin + fout) // 2
        kf_src = os.path.join(OUT, "frames", f"{mid:06d}.png")
        kf_dst = os.path.join(OUT, "keyframes", f"{sid}.png")
        shutil.copyfile(kf_src, kf_dst)
        arr = to_rgb(kf_src)

        attrs = visual_attrs(arr)
        persons = persons_in(arr, characters)

        # 场景聚合：与上一镜主色足够接近就并入同一场景
        blocks = arr.reshape(-1, 3).mean(axis=0) / 255.0
        if prev_blocks is None or float(np.abs(blocks - prev_blocks).mean()) > SCENE_COLOR_THR:
            scene_seq += 1
        scene_id = f"SC{scene_seq:02d}"
        prev_blocks = blocks

        # 台词：按时间区间重叠分配
        t_in, t_out = fin / fps, fout / fps
        lines = []
        for a in asr:
            if a["t_in"] < t_out and a["t_out"] > t_in:
                lines.append({
                    "line_id": f"{sid}-L{len(lines) + 1:02d}",   # 全局唯一，评测才有意义
                    "t_in": a["t_in"], "t_out": a["t_out"],
                    "speaker": a["speaker"], "text": a["text"],
                })

        # 说话人-人物绑定（规则：镜头内在场、占比最大者；无人则未知）
        for ln in lines:
            if persons:
                ln["bound_to"] = persons[0]["char_id"]
                ln["bind_rule"] = "largest_person_in_shot"
            else:
                ln["bound_to"] = None
                ln["bind_rule"] = "off_screen"

        events = sorted({v for kw, v in EVENT_LEXICON.items()
                         if any(kw in ln["text"] for ln in lines)})

        records.append({
            "shot_id": sid,
            "tc_in": tc(fin, fps), "tc_out": tc(fout, fps),
            "frame_in": fin, "frame_out": fout,
            "duration_sec": round((fout - fin) / fps, 2),
            "version_id": "V1-RoughCut",
            "source": {"asset_id": "AST-0001", "uri": "footage/film_01.mp4",
                       "tc_in": tc(fin, fps), "tc_out": tc(fout, fps)},
            "keyframe": f"keyframes/{sid}.png",
            "visual": {**attrs, "characters": [p["char_id"] for p in persons],
                       "tags": f"{attrs['luma_tag']}, {attrs['temp_tag']}, {attrs['sat_tag']}, "
                               f"{'有人物' if persons else '无人物'}"},
            "scene_id": scene_id,
            "characters": [{"char_id": p["char_id"], "name": p["name"], "conf": p["score"]}
                           for p in persons],
            "events": events,
            "dialogue": lines,
            "embedding": None,          # ← 跨模态向量插槽：接 CLIP 后填这里
        })

    # 2) 实体表 + 关系表（生产要素关联）
    ents = {"characters": {}, "scenes": {}, "events": set(), "dialogues": [],
            "versions": {"V1-RoughCut": {"label": "粗剪版 v1", "assets": ["AST-0001"]}},
            "shots": [r["shot_id"] for r in records]}
    rels = []
    for r in records:
        for c in r["characters"]:
            ents["characters"][c["char_id"]] = {"name": c["name"]}
            rels.append((r["shot_id"], "has_character", c["char_id"]))
        ents["scenes"].setdefault(r["scene_id"], {"shots": []})["shots"].append(r["shot_id"])
        rels.append((r["shot_id"], "belongs_to_scene", r["scene_id"]))
        rels.append((r["shot_id"], "instance_of_version", r["version_id"]))
        for e in r["events"]:
            ents["events"].add(e)
            rels.append((r["shot_id"], "depicts_event", e))
        for ln in r["dialogue"]:
            ents["dialogues"].append(ln)
            rels.append((r["shot_id"], "has_dialogue", ln["line_id"]))
            if ln["bound_to"]:
                rels.append((ln["line_id"], "spoken_by", ln["bound_to"]))
    for a, b in zip(records, records[1:]):
        rels.append((a["shot_id"], "followed_by", b["shot_id"]))

    ents["events"] = sorted(ents["events"])
    with open(os.path.join(OUT, "structure.json"), "w", encoding="utf-8") as fp:
        json.dump({"fps": fps, "version_id": "V1-RoughCut", "shots": records},
                  fp, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT, "entities.json"), "w", encoding="utf-8") as fp:
        json.dump(ents, fp, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT, "relations.csv"), "w", encoding="utf-8-sig", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["head", "relation", "tail"])
        w.writerows(rels)

    n_line = sum(len(r["dialogue"]) for r in records)
    print(f"[structure] {len(records)} 个镜头 → {len(ents['scenes'])} 个场景, "
          f"{len(ents['characters'])} 个人物, {len(ents['events'])} 类事件, {n_line} 条台词, "
          f"{len(rels)} 条关系")


if __name__ == "__main__":
    main()
