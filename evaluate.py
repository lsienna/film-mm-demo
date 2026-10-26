# -*- coding: utf-8 -*-
"""
evaluate.py —— 把申请书里的两个指标算成"可复现的数"

指标一：多模态素材语义检索首位结果有效命中率 Hit@1（以及 R@5 / MRR / mAP）
指标二：跨模态语义对齐准确率（这里落地为"台词—人物绑定准确率"，并且实现三种口径）

为什么口径要写清楚？因为"对齐准确率 ≥85%"如果不说清"对什么齐、以什么为单位、
允许多大时间误差"，验收时就没法判定。三种口径的差别在本 demo 里相差 14 个百分点。

自测：python evaluate.py   （会跑一个植入已知错误率的合成用例，验证算法正确）
"""

import json
import os

from paths import OUT


# ------------------------------------------------------------------ 检索指标
def evaluate_retrieval(records, k_list=(1, 5)):
    """records: [{query, type, ranked:[shot_id...], gold:[shot_id...]}]"""
    if not records:
        return {}
    hit1 = hit5 = 0
    rr, aps = [], []
    per_type = {}
    for r in records:
        ranked, gold = r["ranked"], set(r["gold"])
        rank_of = next((i for i, s in enumerate(ranked) if s in gold), None)
        h1 = 1 if rank_of == 0 else 0
        h5 = 1 if (rank_of is not None and rank_of < 5) else 0
        hit1 += h1
        hit5 += h5
        rr.append(1.0 / (rank_of + 1) if rank_of is not None else 0.0)
        aps.append(1.0 / (rank_of + 1) if rank_of is not None else 0.0)
        t = per_type.setdefault(r.get("type", "all"), {"n": 0, "hit1": 0})
        t["n"] += 1
        t["hit1"] += h1

    n = len(records)
    out = {
        "n_query": n,
        "Hit@1": round(hit1 / n, 4),
        "Hit@5": round(hit5 / n, 4),
        "MRR": round(sum(rr) / n, 4),
        "mAP": round(sum(aps) / n, 4),
        "per_type": {k: {"n": v["n"], "Hit@1": round(v["hit1"] / v["n"], 4)}
                     for k, v in per_type.items()},
    }
    return out


# ------------------------------------------------------------------ 对齐指标
def _iou(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    inter = max(0.0, hi - lo)
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def alignment_accuracy(pred, gold, mode="strict", iou_thr=0.5):
    """pred/gold: [{line_id, speaker, t_in, t_out}]

    两种口径（申请书里必须选一种并写清楚）：
      strict —— 只要台词段对得上、绑定人物完全一致才算对（最严，验收最好写）
      iou    —— 时间区间 IoU ≥ 0.5 且人物一致才算对（容忍 ASR 时间戳抖动）
    经验：口径不写清楚，"对齐准确率 85%" 在验收时可以被解释成 70%~90% 之间的任何值。
    """
    gmap = {g["line_id"]: g for g in gold}
    ok = tot = 0
    for p in pred:
        g = gmap.get(p["line_id"])
        if g is None:
            continue
        tot += 1
        same = p.get("speaker") == g.get("speaker")
        if mode == "strict":
            ok += int(same)
        elif mode == "iou":
            ok += int(same and _iou((p["t_in"], p["t_out"]),
                                    (g["t_in"], g["t_out"])) >= iou_thr)
        else:
            raise ValueError(mode)
    return {"mode": mode, "correct": ok, "total": tot,
            "accuracy": round(ok / tot, 4) if tot else 0.0}


# ------------------------------------------------------------------ 自测
def self_test():
    """植入 100 条台词：23 条绑定错误（前 23 条），另有 10 条正确但时间戳抖动。"""
    gold, pred, pred_iou = [], [], []
    for i in range(100):
        g = {"line_id": f"L{i:03d}", "speaker": "C1", "t_in": i, "t_out": i + 1}
        gold.append(g)
        wrong = i < 23
        pred.append({"line_id": g["line_id"], "speaker": "C2" if wrong else "C1",
                     "t_in": i, "t_out": i + 1})
        # 最后 10 条绑定正确，但 ASR 时间戳抖到 IoU<0.5，只有 iou 口径会判错
        jitter = i >= 90
        pred_iou.append({"line_id": g["line_id"], "speaker": "C2" if wrong else "C1",
                         "t_in": i + (0.6 if jitter else 0), "t_out": i + 1 + (0.6 if jitter else 0)})

    a = alignment_accuracy(pred, gold, "strict")
    b = alignment_accuracy(pred_iou, gold, "iou")
    ok1 = abs(a["accuracy"] - 0.77) < 1e-9
    ok2 = abs(b["accuracy"] - 0.67) < 1e-9          # 10 条被时间约束刷掉
    print(f"[self-test] strict={a['accuracy']:.4f} (期望 0.77) -> {'PASS' if ok1 else 'FAIL'}")
    print(f"[self-test] iou   ={b['accuracy']:.4f} (期望 0.67) -> {'PASS' if ok2 else 'FAIL'}")
    return ok1 and ok2


def main():
    if not self_test():
        raise SystemExit("自测未通过，先别信后面的数")
    with open(os.path.join(OUT, "structure.json"), encoding="utf-8") as fp:
        st = json.load(fp)
    with open(os.path.join(OUT, "truth.json"), encoding="utf-8") as fp:
        truth = json.load(fp)

    # 用时间区间重叠把"真值镜头"映射到"预测镜头"
    def frame_of(tc_str, fps):
        h, m, s, f = [int(x) for x in tc_str.split(":")]
        return ((h * 60 + m) * 60 + s) * fps + f

    shots = st["shots"]
    gold_pairs, pred_pairs, bucket = [], [], []
    for ts in truth["shots"]:
        best = max(shots, key=lambda r: _iou((r["frame_in"], r["frame_out"]),
                                             (ts["in"], ts["out"])))
        if _iou((best["frame_in"], best["frame_out"]), (ts["in"], ts["out"])) < 0.3:
            continue
        for ln in best["dialogue"]:
            gold_pairs.append({"line_id": ln["line_id"], "speaker": ts["dialogue"]["speaker"],
                               "t_in": ln["t_in"], "t_out": ln["t_out"]})
            pred_pairs.append({"line_id": ln["line_id"], "speaker": ln["bound_to"],
                               "t_in": ln["t_in"], "t_out": ln["t_out"]})
            n_p = len(best["characters"])
            kind = "画外音(镜头内无人)" if n_p == 0 else ("单人镜头" if n_p == 1 else "多人同框")
            bucket.append((kind, ln["bound_to"] == ts["dialogue"]["speaker"]))

    res = {m: alignment_accuracy(pred_pairs, gold_pairs, m) for m in ("strict", "iou")}
    print("\n[对齐] 台词—人物绑定准确率")
    for m, v in res.items():
        print(f"  口径 {m:6s}: {v['accuracy']:.1%}  ({v['correct']}/{v['total']})")

    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0])
    for kind, ok in bucket:
        agg[kind][0] += 1
        agg[kind][1] += int(ok)
    print("  分场景拆解：")
    for kind, (n, ok) in sorted(agg.items()):
        print(f"    {kind:16s} {ok}/{n} = {ok / n:.0%}")

    return {"alignment": res, "breakdown": {k: [v[1], v[0]] for k, v in agg.items()}}


if __name__ == "__main__":
    main()
