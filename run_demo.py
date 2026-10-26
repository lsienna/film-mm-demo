# -*- coding: utf-8 -*-
"""
run_demo.py —— 一键跑通：合成素材 → 镜头切分 → 结构化 → 检索评测 → 报告

    python run_demo.py

不需要 GPU、不需要联网、不需要装 torch。跨模态分支（B2/CLIP）在缺少依赖时
自动跳过，并在报告里标 SKIP —— 这正是你拿到 GPU 机器后要补的第一件事。
"""

import json
import os
import sys

import numpy as np

import make_footage
import shot_detect
import structure
import evaluate
import retrieval
from paths import OUT


def _setup_console():
    """Windows 控制台默认是 cp936，打印 ✓/✗ 会抛 UnicodeEncodeError 直接崩。
    这里把编码无法表示的字符降级成 '?'，中文不受影响，也不改控制台代码页。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass


_setup_console()

# 查询集：4 种难度，每种都要有"唯一正解"，否则 Hit@1 无意义
QUERIES = [
    # --- T1 台词原句（传统媒资系统的强项：ASR 全文检索）
    ("把高光收回来", "S002", "T1-台词原句"),
    ("今晚必须交片", "S003", "T1-台词原句"),
    ("第九场重剪", "S004", "T1-台词原句"),
    ("那年也是这样的雨", "S005", "T1-台词原句"),
    # --- T2 人物组合（需要"人物在场"这一结构化要素）
    ("林晓和周航同时出现在画面里", "S005", "T2-人物组合"),
    ("只有周航一个人出现", "S003", "T2-人物组合"),
    ("白天的空镜，画面里没有人物", "S004", "T2-人物组合"),
    # --- T3 视觉属性（需要低层视觉标签，是 B1 与 B2 的交界）
    ("低饱和度的夜晚，画面里没有人物", "S007", "T3-视觉属性"),
    ("中饱和度的夜晚里有人物", "S003", "T3-视觉属性"),
    # --- T4 语义泛化（改写、无字面重合，必须靠跨模态语义）
    ("一对恋人多年后重逢，气氛伤感", "S005", "T4-语义泛化"),
    ("剧组赶工的压力", "S003", "T4-语义泛化"),
    ("画面曝光过度引发的不满", "S001", "T4-语义泛化"),
    ("杀青时的告别", "S007", "T4-语义泛化"),
]


def iou(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    return max(0.0, hi - lo) / max(1e-6, max(a[1], b[1]) - min(a[0], b[0]))


def map_gold_to_pred(truth, shots):
    """把真值镜头 id 映射到预测镜头 id（真实系统里就是标注与流水线输出的对齐）。"""
    m = {}
    for ts in truth["shots"]:
        best = max(shots, key=lambda r: iou((r["frame_in"], r["frame_out"]), (ts["in"], ts["out"])))
        if iou((best["frame_in"], best["frame_out"]), (ts["in"], ts["out"])) >= 0.3:
            m[ts["shot_id"]] = best["shot_id"]
    return m


def main():
    print("=" * 72)
    print("Step 1/5  合成影视测试片")
    print("=" * 72)
    make_footage.main()

    print("\n" + "=" * 72)
    print("Step 2/5  镜头边界检测")
    print("=" * 72)
    shot_detect.main()
    with open(os.path.join(OUT, "shots.json"), encoding="utf-8") as fp:
        shot_doc = json.load(fp)
    shot_eval = shot_doc["eval"]

    print("\n" + "=" * 72)
    print("Step 3/5  镜头级结构化 + 生产要素关联")
    print("=" * 72)
    structure.main()

    print("\n" + "=" * 72)
    print("Step 4/5  语义检索：B1 结构化召回 / B2 跨模态召回 / B3 融合")
    print("=" * 72)
    with open(os.path.join(OUT, "structure.json"), encoding="utf-8") as fp:
        st = json.load(fp)
    with open(os.path.join(OUT, "truth.json"), encoding="utf-8") as fp:
        truth = json.load(fp)
    shots = st["shots"]
    ids = [s["shot_id"] for s in shots]
    docs = [retrieval.shot_document(s) for s in shots]
    bm25 = retrieval.BM25(docs)

    # B2：跨模态编码器（无依赖则跳过）
    enc = retrieval.CLIPEncoder()
    b2_scores = None
    if enc.ok:
        kf = [os.path.join(OUT, s["keyframe"]) for s in shots]
        vt = enc.encode_images(kf)
        qt = enc.encode_texts([q[0] for q in QUERIES])
        vt = vt / (np.linalg.norm(vt, axis=1, keepdims=True) + 1e-9)
        qt = qt / (np.linalg.norm(qt, axis=1, keepdims=True) + 1e-9)
        b2_scores = qt @ vt.T
        print(f"跨模态编码器：{enc.reason}")
    else:
        print(f"跨模态编码器：{enc.reason}")

    gold_map = map_gold_to_pred(truth, shots)
    rec_b1, rec_b1s, rec_b3 = [], [], []
    rows = []
    for qi, (q, gold_truth, qtype) in enumerate(QUERIES):
        gold = gold_map.get(gold_truth)
        s1 = bm25.rank(q, docs)
        order1 = [ids[i] for i in sorted(range(len(ids)), key=lambda i: -s1[i])]

        # B1s：先按硬约束砍候选，再在候选内用 BM25 排序
        cond = retrieval.understand(q, shots)
        keep, fallback = retrieval.apply_conditions(cond, shots)
        s1s = [s1[i] if ids[i] in keep else -1e9 for i in range(len(ids))]
        order_sf = [ids[i] for i in sorted(range(len(ids)), key=lambda i: -s1s[i])]

        if b2_scores is not None:
            s3 = retrieval.fuse(np.asarray(s1s), list(b2_scores[qi]), w=0.6)
        else:
            s3 = retrieval.fuse(np.asarray(s1s), None)
        order3 = [ids[i] for i in sorted(range(len(ids)), key=lambda i: -s3[i])]

        rec_b1.append({"query": q, "type": qtype, "ranked": order1, "gold": [gold]})
        rec_b1s.append({"query": q, "type": qtype, "ranked": order_sf, "gold": [gold]})
        rec_b3.append({"query": q, "type": qtype, "ranked": order3, "gold": [gold]})
        rows.append((qtype, q, gold, order1[0], order_sf[0], order3[0],
                     order1[0] == gold, order_sf[0] == gold))

    r1 = evaluate.evaluate_retrieval(rec_b1)
    r1s = evaluate.evaluate_retrieval(rec_b1s)
    r3 = evaluate.evaluate_retrieval(rec_b3)

    print(f"\n{'类型':<12}{'查询':<24}{'正解':<6}{'B1':<6}{'B1s':<6}{'B3':<6}{'B1s命中'}")
    print("-" * 78)
    for t, q, g, t1, t1s, t3, ok1, oks in rows:
        print(f"{t:<12}{q:<24}{g:<6}{t1:<6}{t1s:<6}{t3:<6}{'✓' if oks else '✗'}")

    def line(name, r):
        return (f"  {name:<22} Hit@1={r['Hit@1']:.1%}  Hit@5={r['Hit@5']:.1%}  "
                f"MRR={r['MRR']:.3f}")

    print("\n" + "=" * 72)
    print("Step 5/5  指标汇总")
    print("=" * 72)
    print(line("B1  纯文本召回(BM25)", r1))
    print(line("B1s 结构化约束+BM25", r1s))
    if b2_scores is None:
        print(f"  {'B2  跨模态(CLIP)':<22} SKIP —— {enc.reason}")
    else:
        print(line("B2  跨模态(CLIP)", evaluate.evaluate_retrieval(
            [{"query": q, "type": t, "gold": [gold_map.get(g)],
              "ranked": [ids[i] for i in np.argsort(-b2_scores[qi])]}
             for qi, (q, g, t) in enumerate(QUERIES)])))
    print(line("B3  融合(无B2时=B1s)", r3))
    print("\n  分类型 Hit@1（B1s）：")
    for k, v in sorted(r1s["per_type"].items()):
        print(f"    {k:<16} n={v['n']:<3} Hit@1={v['Hit@1']:.0%}")

    al = evaluate.main()

    # ---------------- 写报告 ----------------
    hc, dv = shot_eval["hard_cut"], shot_eval["dissolve"]
    md = []
    md.append("# 影视素材跨模态语义理解 Demo 报告\n")
    md.append("> 全流程由 `run_demo.py` 自动生成，纯 numpy 实现，无需 GPU / 联网。")
    md.append("> **注意：本 demo 素材是合成的，数值只用于验证「管线与指标口径」，"
              "不代表真实影视数据上的性能。**\n")
    md.append("## 1. 镜头切分\n")
    md.append("| 边界类型 | Precision | Recall | F1 |\n|---|---|---|---|")
    md.append(f"| 硬切 | {hc['precision']:.3f} | {hc['recall']:.3f} | {hc['f1']:.3f} |")
    md.append(f"| 叠化（渐变） | {dv['precision']:.3f} | {dv['recall']:.3f} | {dv['f1']:.3f} |")
    md.append(f"\n共切出 {shot_doc['n_shots']} 个镜头（真值 {shot_eval['n_shots_gt']} 个）。\n")
    md.append("## 2. 检索指标\n")
    md.append("| 检索器 | Hit@1 | Hit@5 | MRR |\n|---|---|---|---|")
    md.append(f"| B1 纯文本召回（BM25 on 台词+属性标签） | {r1['Hit@1']:.1%} | {r1['Hit@5']:.1%} | {r1['MRR']:.3f} |")
    md.append(f"| B1s 结构化约束 + BM25 | {r1s['Hit@1']:.1%} | {r1s['Hit@5']:.1%} | {r1s['MRR']:.3f} |")
    md.append(f"| B2 跨模态 CLIP | SKIP | SKIP | SKIP |")
    md.append(f"| B3 融合（无 B2 时 = B1s） | {r3['Hit@1']:.1%} | {r3['Hit@5']:.1%} | {r3['MRR']:.3f} |")
    md.append("\n> 查询集 13 条、镜头 7 个，**样本量太小，百分比只用于验证口径，不能当性能结论**。\n")
    md.append("### 分类型 Hit@1（B1s）\n")
    md.append("| 查询类型 | 条数 | Hit@1 |\n|---|---|---|")
    for k, v in sorted(r1s["per_type"].items()):
        md.append(f"| {k} | {v['n']} | {v['Hit@1']:.0%} |")
    md.append("\n## 3. 对齐指标（台词—人物绑定）\n")
    md.append("| 口径 | 准确率 |\n|---|---|")
    for m, v in al["alignment"].items():
        md.append(f"| {m} | {v['accuracy']:.1%} ({v['correct']}/{v['total']}) |")
    md.append("\n| 场景类型 | 正确/总数 |\n|---|---|")
    for k, (ok, n) in sorted(al["breakdown"].items()):
        md.append(f"| {k} | {ok}/{n} |")
    worst = min(r1s["per_type"].items(), key=lambda kv: kv[1]["Hit@1"])
    best = max(r1s["per_type"].items(), key=lambda kv: kv[1]["Hit@1"])
    md.append("\n## 4. 结论\n")
    md.append(f"1. 纯文本召回（B1）Hit@1 只有 {r1['Hit@1']:.0%}；加上结构化硬约束（B1s）升到 "
              f"{r1s['Hit@1']:.0%}。**结构化要素本身就有可量化的检索收益**，这是申请书里"
              "\"生产要素关联\"最硬的论据。")
    md.append(f"2. 最强的是「{best[0]}」（{best[1]['Hit@1']:.0%}），最弱的是「{worst[0]}」"
              f"（{worst[1]['Hit@1']:.0%}）。短板是改写/泛化类查询——它们的字面与素材文本"
              "毫无重合，只能靠跨模态语义表示解决，而这正是指南 80% 指标的关键。")
    md.append(f"3. 台词—人物绑定准确率（strict 口径）="
              f"{al['alignment']['strict']['accuracy']:.0%}，缺口集中在画外音与多人同框："
              "画外音时镜头里根本没有人脸可绑，必须引入声纹+唇动+剧本先验。")
    md.append("4. 距离指南指标还差多少：本 demo 只有 13 条查询、7 个镜头，不能外推。"
              "但方向已经清楚——**B2 换成 Chinese-CLIP / InternVideo2 之后，同一套脚本直接出对比表**。")
    md.append("5. 指标口径已经写成函数（`evaluate.py`，含自测），申请书里的考核指标可以直接引用。")
    with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as fp:
        fp.write("\n".join(md) + "\n")
    print(f"\n报告已写出：{os.path.join(OUT, 'report.md')}")


if __name__ == "__main__":
    main()
