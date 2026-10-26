# -*- coding: utf-8 -*-
"""
retrieval.py —— 语义检索的两条召回通路 + 融合

B1 结构化/文本召回（本地可跑，真数据）：
    把镜头记录拍平成"文档"（台词 + 视觉属性标签 + 人物名 + 事件），
    中文按字 bigram 切词，用 BM25 打分。这就是传统媒资系统的检索方式。

B2 跨模态召回（需要 CLIP 权重，本地无 GPU/无网时自动跳过）：
    文本塔 + 视觉塔，余弦相似度。接口留好，换模型只改 CLIPEncoder。

B3 融合：normalize 后加权求和，对应申请书里"结构化约束 + 跨模态语义 + 重排"的技术路线。

跑法见 run_demo.py；单独自测：python retrieval.py
"""

import math
import os
import re

# ---------------------------------------------------------------- 分词（中文友好）
# 只用字 bigram，不用单字：单字会把"这个"里的"个"送进倒排，制造跨语义的假匹配。
_STOP = {"画面", "一个", "这个", "那个", "什么", "怎么", "我们", "你们", "他们"}


def tokenize(text):
    """英文/数字整词 + 中文按"连续汉字段"切 bigram（段内切，不跨标点）。"""
    text = (text or "").lower()
    toks = re.findall(r"[a-z0-9]+", text)
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        toks += [run[i:i + 2] for i in range(len(run) - 1)]
    return [t for t in toks if t not in _STOP]


def shot_document(rec):
    """把一条镜头结构化记录拍平成可检索文本（这就是"知识化组织"的检索视图）。"""
    parts = []
    for ln in rec.get("dialogue", []):
        parts.append(ln.get("text", ""))
    v = rec.get("visual", {})
    parts.append(v.get("luma_tag", ""))
    parts.append(v.get("temp_tag", ""))
    parts.append(v.get("sat_tag", ""))
    parts.append(v.get("tags", ""))
    parts += [c.get("name", "") for c in rec.get("characters", [])]
    parts += rec.get("events", [])
    if not rec.get("characters"):
        parts.append("无人物 空镜")
    else:
        parts.append("有人物")
    return " ".join(p for p in parts if p)


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [tokenize(d) for d in docs]
        self.k1, self.b = k1, b
        self.N = len(self.docs)
        self.avgdl = (sum(len(d) for d in self.docs) / self.N) if self.N else 0.0
        df = {}
        for d in self.docs:
            for t in set(d):
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log(1 + (self.N - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.tf = []
        for d in self.docs:
            f = {}
            for t in d:
                f[t] = f.get(t, 0) + 1
            self.tf.append(f)

    def score(self, query, i):
        q = tokenize(query)
        dl = len(self.docs[i]) or 1
        s = 0.0
        for t in q:
            if t not in self.idf:
                continue
            f = self.tf[i].get(t, 0)
            if not f:
                continue
            s += self.idf[t] * f * (self.k1 + 1) / (
                f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
        return s

    def rank(self, query, docs):
        sc = [self.score(query, i) for i in range(len(docs))]
        return sc


class CLIPEncoder:
    """跨模态编码器插槽。有权重就跑，没有就返回 None（下游自动降级）。

    真实用法（拿到 GPU 机器后）：
        pip install torch transformers ftfy
        # 中文推荐 OFA-Sys/chinese-clip-vit-large-patch16，英文可用 openai/clip-vit-large-patch14
    """

    def __init__(self, model_name="OFA-Sys/chinese-clip-vit-base-patch16"):
        self.ok, self.reason = False, ""
        try:
            import torch                      # noqa: F401
            from transformers import CLIPModel, CLIPProcessor   # noqa: F401
            self.torch = torch
            self.model = CLIPModel.from_pretrained(model_name).eval()
            self.proc = CLIPProcessor.from_pretrained(model_name)
            self.ok, self.reason = True, model_name
        except Exception as e:                # 没装/没网/没权重
            self.reason = f"跨模态编码器不可用（{type(e).__name__}），已跳过 B2"

    def encode_texts(self, texts):
        with self.torch.no_grad():
            inp = self.proc(text=texts, return_tensors="pt", padding=True, truncation=True)
            return self.model.get_text_features(**inp).cpu().numpy()

    def encode_images(self, paths):
        from PIL import Image
        with self.torch.no_grad():
            imgs = [Image.open(p).convert("RGB") for p in paths]
            inp = self.proc(images=imgs, return_tensors="pt")
            return self.model.get_image_features(**inp).cpu().numpy()


def norm(x):
    import numpy as np
    x = np.asarray(x, dtype="float32")
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else x * 0.0


# ---------------------------------------------------------- 查询理解 + 结构化约束
# 这一步就是指南里"生产要素关联"带来的红利：
# 把"只有周航一个人"翻译成 (人物=周航) AND (镜头内人数==1)，
# 直接砍掉候选集，纯向量检索做不到这种硬约束。
def understand(q, shots):
    names = sorted({c["name"] for s in shots for c in s["characters"]})
    cond = {"names": [n for n in names if n in q], "n_person": None,
            "luma": None, "temp": None, "sat": None,
            "negative": any(k in q for k in ("空镜", "没有人物", "没有人", "无人"))}
    if cond["negative"]:
        cond["n_person"] = ("==", 0)
    elif any(k in q for k in ("同时", "一起", "两人", "两个人")):
        cond["n_person"] = (">=", 2)
    elif any(k in q for k in ("一个人", "独自", "单独")):
        cond["n_person"] = ("==", 1)
    for w in ("白天", "黄昏", "昏暗", "夜晚"):
        if w in q:
            cond["luma"] = w
            break
    for w in ("暖色调", "冷色调", "中性色"):
        if w in q:
            cond["temp"] = w
            break
    for w in ("高饱和", "中饱和", "低饱和"):
        if w in q:
            cond["sat"] = w
            break
    return cond


def apply_conditions(cond, shots):
    """返回满足硬约束的镜头 id 列表；若过滤后为空则退化为全量（并标记）。"""
    keep = []
    for s in shots:
        n = len(s.get("characters", []))
        v = s.get("visual", {})
        names = {c["name"] for c in s.get("characters", [])}
        if cond["names"] and not set(cond["names"]).issubset(names):
            continue
        if cond["n_person"] is not None:
            op, val = cond["n_person"]
            if (op == "==" and n != val) or (op == ">=" and n < val):
                continue
        if cond["luma"] and v.get("luma_tag") != cond["luma"]:
            continue
        if cond["temp"] and v.get("temp_tag") != cond["temp"]:
            continue
        if cond["sat"] and v.get("sat_tag") != cond["sat"]:
            continue
        keep.append(s["shot_id"])
    return keep, (not keep)


def fuse(b1, b2=None, w=0.5):
    """融合：B1 与 B2 各自 min-max 归一化后加权。没有 B2 时退化为 B1。"""
    if b2 is None:
        return norm(b1)
    return (1 - w) * norm(b1) + w * norm(b2)


if __name__ == "__main__":
    docs = ["这个版本的调色我觉得还是太亮了 高亮日景 暖调 有人 林",
            "夜里空无一人的天台 夜景 冷调 无人"]
    bm = BM25(docs)
    print("BM25 自测：",
          [round(s, 3) for s in bm.rank("调色 太亮", docs)], "(第 0 条应更高)")
