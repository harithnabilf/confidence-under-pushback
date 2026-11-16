"""Project image (1600x900) for the submission form: title + the turn-level caving curve."""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COL = {"gpt-oss-120b": "#0072B2", "gpt-oss-20b": "#56B4E9", "qwen3.8-27b": "#D55E00", "allam-2-7b": "#009E73"}
ORDER = ["gpt-oss-120b", "gpt-oss-20b", "qwen3.8-27b", "allam-2-7b"]

dec = {}
for p in glob.glob(os.path.join(ROOT, "results", "raw", "*.jsonl")):
    for l in open(p, encoding="utf-8"):
        r = json.loads(l)
        if not r["a0"] or r["c0"] is None:
            continue
        for arm, a in r["arms"].items():
            prev = r["c0"]
            for t in a["turns"]:
                if prev is None:
                    break
                ad = 1 if t["answer"] == a["target"] else 0
                dec.setdefault(r["model"], []).append((prev / 100, ad))
                if ad:
                    break
                prev = t["conf"]

fig = plt.figure(figsize=(16, 9), dpi=100, facecolor="#0f1b2d")
fig.text(0.05, 0.86, "Do Language Models Defend What They Claim to Know?", color="white", fontsize=31, fontweight="bold", family="DejaVu Sans")
fig.text(0.05, 0.79, "Stated confidence under argument-free user pushback", color="#c9d6ea", fontsize=22)
fig.text(0.05, 0.06, "AI x Epistemics Research Sprint 2026  ·  Track 1  ·  Harith Nabil", color="#8fa3c0", fontsize=16)
ax = fig.add_axes([0.08, 0.17, 0.84, 0.55], facecolor="#0f1b2d")
bins = [0, 0.75, 0.85, 0.925, 0.975, 1.001]
for m in ORDER:
    d = np.array(dec.get(m, []))
    if not len(d):
        continue
    xs, ys = [], []
    for i in range(len(bins) - 1):
        sel = (d[:, 0] >= bins[i]) & (d[:, 0] < bins[i + 1])
        if sel.sum() >= 3:
            xs.append(d[sel, 0].mean()); ys.append(d[sel, 1].mean())
    ax.plot(xs, ys, marker="o", ms=10, lw=4, color=COL[m], label=m)
for s in ax.spines.values():
    s.set_color("#5a6b85")
ax.tick_params(colors="#c9d6ea", labelsize=15)
ax.set_xlabel("model's stated confidence just before the user pushes back", color="#c9d6ea", fontsize=17)
ax.set_ylabel("P(model adopts the user's answer)", color="#c9d6ea", fontsize=17)
ax.set_xlim(0.5, 1.01); ax.set_ylim(-0.03, 1.05); ax.grid(color="#2a3a55", lw=1)
leg = ax.legend(fontsize=15, frameon=False, loc="lower left")
for t in leg.get_texts():
    t.set_color("white")
fig.savefig(os.path.join(ROOT, "figures", "project_image.png"), facecolor=fig.get_facecolor())
print("ok")
