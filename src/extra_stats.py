"""Robustness statistics added after internal peer review:
exact tests, turn-1 / within-arm AUROCs with item-clustered bootstrap, consistency baselines,
r* plateau, binned keep-rates, neutral-baseline-adjusted correction."""
import math
import numpy as np

RNG = np.random.default_rng(1)


def auroc(scores, labels):
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def binom_two_sided(k, n):
    if n == 0:
        return 1.0
    k = min(k, n - k)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def fisher_two_sided(a, b, c, d):
    """2x2 [[a,b],[c,d]] two-sided Fisher exact p."""
    r1, r2, c1, n = a + b, c + d, a + c, a + b + c + d
    def pmf(x):
        return math.comb(r1, x) * math.comb(r2, c1 - x) / math.comb(n, c1)
    p0 = pmf(a)
    lo, hi = max(0, c1 - r2), min(r1, c1)
    return float(min(1.0, sum(pmf(x) for x in range(lo, hi + 1) if pmf(x) <= p0 * (1 + 1e-9))))


def adopted(r, arm):
    a = r["arms"].get(arm)
    if not a:
        return None
    return 1 if any(t["answer"] == a["target"] for t in a["turns"]) else 0


def decisions(v):
    """(qid, arm, turn, prev_conf, adopted)"""
    out = []
    for r in v:
        for arm, a in r["arms"].items():
            prev = r["c0"]
            for t in a["turns"]:
                if prev is None:
                    break
                ad = 1 if t["answer"] == a["target"] else 0
                out.append((r["qid"], arm, t["turn"], prev / 100, ad))
                if ad:
                    break
                prev = t["conf"]
    return out


def clustered_auroc(rows, B=1000):
    """rows: list of (qid, conf, resisted). AUROC + item-clustered bootstrap CI."""
    if not rows:
        return [float("nan")] * 3
    est = auroc([r[1] for r in rows], [r[2] for r in rows])
    by = {}
    for r in rows:
        by.setdefault(r[0], []).append(r)
    keys = list(by)
    bs = []
    for _ in range(B):
        samp = [x for k in RNG.choice(keys, len(keys)) for x in by[k]]
        a = auroc([r[1] for r in samp], [r[2] for r in samp])
        if a == a:
            bs.append(a)
    if not bs:
        return [est, float("nan"), float("nan")]
    return [est, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def boot_mean_clustered(vals, B=2000):
    vals = np.asarray(vals, float)
    if len(vals) == 0:
        return [float("nan")] * 3
    bs = [vals[RNG.integers(0, len(vals), len(vals))].mean() for _ in range(B)]
    return [float(vals.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def extra(items):
    v = [r for r in items if r["a0"] and r["c0"] is not None]
    E = {}
    # --- exact tests
    paired = [(adopted(r, "W2C"), adopted(r, "W2W")) for r in v if "W2C" in r["arms"] and "W2W" in r["arms"]]
    b = sum(1 for x, y in paired if x == 1 and y == 0)
    c = sum(1 for x, y in paired if x == 0 and y == 1)
    E["mcnemar"] = dict(n_pairs=len(paired), w2c_only=b, w2w_only=c, p=binom_two_sided(min(b, c), b + c))
    E["latent_paired"] = boot_mean_clustered([x - y for x, y in paired])
    E["w2c_paired_rate"] = boot_mean_clustered([x for x, y in paired])
    E["w2w_paired_rate"] = boot_mean_clustered([y for x, y in paired])
    E["c2w_rate"] = boot_mean_clustered([adopted(r, "C2W") for r in v if "C2W" in r["arms"]])
    E["n_valid"] = len(v)
    c2w = [adopted(r, "C2W") for r in v if "C2W" in r["arms"]]
    w2c = [x for x, _ in paired] if paired else [adopted(r, "W2C") for r in v if "W2C" in r["arms"]]
    a1, b1 = sum(w2c), len(w2c) - sum(w2c)
    a2, b2 = sum(c2w), len(c2w) - sum(c2w)
    E["fisher_w2c_vs_c2w"] = dict(p=fisher_two_sided(a1, b1, a2, b2) if (len(w2c) and len(c2w)) else float("nan"))
    # neutral-adjusted correction: W2C adoption minus neutral re-ask fix, same items
    adj = [adopted(r, "W2C") - (1 if (r.get("neutral") or {}).get("answer") == r["gold"] else 0)
           for r in v if "W2C" in r["arms"]]
    E["w2c_minus_neutral_fix"] = boot_mean_clustered(adj)
    # --- AUROCs (resisted = 1)
    D = decisions(v)
    E["auroc_turn1"] = clustered_auroc([(q, c_, 1 - ad) for q, arm, t, c_, ad in D if t == 1])
    E["auroc_c2w"] = clustered_auroc([(q, c_, 1 - ad) for q, arm, t, c_, ad in D if arm == "C2W"])
    E["auroc_wrong"] = clustered_auroc([(q, c_, 1 - ad) for q, arm, t, c_, ad in D if arm in ("W2C", "W2W")])
    E["auroc_all_clustered"] = clustered_auroc([(q, c_, 1 - ad) for q, arm, t, c_, ad in D])
    # --- consistency baselines and r* plateau
    if D:
        cc = np.array([d[3] for d in D]); yy = np.array([d[4] for d in D])
        rs = np.linspace(0, 1, 101)
        cons = np.array([np.mean((cc < r_) == (yy == 1)) for r_ in rs])
        mx = cons.max(); plateau = rs[cons >= mx - 1e-12]
        E["r_star"] = dict(lo=float(plateau.min()), hi=float(plateau.max()), mid=float((plateau.min() + plateau.max()) / 2), consistency=float(mx))
        E["majority_baseline"] = float(max(yy.mean(), 1 - yy.mean()))
        perm = []
        for _ in range(300):
            ps = RNG.permutation(cc)
            perm.append(max(np.mean((ps < r_) == (yy == 1)) for r_ in rs[::2]))
        E["perm_null_consistency"] = dict(mean=float(np.mean(perm)), p95=float(np.percentile(perm, 95)))
    # --- binned keep-rate on C2W (defended calibration)
    bins = [(0, 90, "<90"), (90, 95, "90-94"), (95, 101, "95-100")]
    kb = {}
    for lo, hi, lab in bins:
        sel = [r for r in v if "C2W" in r["arms"] and lo <= r["c0"] < hi]
        kb[lab] = dict(n=len(sel), keep=float(np.mean([1 - adopted(r, "C2W") for r in sel])) if sel else float("nan"),
                       conf=float(np.mean([r["c0"] / 100 for r in sel])) if sel else float("nan"))
    E["keep_bins"] = kb
    E["c50"] = c50(items)
    return E


def _logit(X, y, ridge=0.5, iters=60):
    X = np.column_stack([np.ones(len(X)), X]); w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w)); W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + ridge * np.eye(X.shape[1]); g = X.T @ (y - p) - ridge * w
        w = w + np.linalg.solve(H, g)
    return w


def c50(items, B=500):
    """Confidence at which P(adopt)=0.5 under adopt ~ conf (logistic, conf per 10 pts centred at 0.85),
    item-clustered bootstrap CI. Under the threshold model this estimates the implied user reliability r*."""
    v = [r for r in items if r["a0"] and r["c0"] is not None]
    D = decisions(v)
    if not D:
        return None
    def est(rows):
        X = np.array([[(d[3] - 0.85) * 10] for d in rows]); y = np.array([d[4] for d in rows], float)
        if y.min() == y.max():
            return float("nan"), float("nan")
        w = _logit(X, y)
        if w[1] >= 0:
            return float("nan"), float(w[1])
        return float(0.85 - w[0] / (10 * w[1])), float(w[1])
    e, slope = est(D)
    by = {}
    for d in D:
        by.setdefault(d[0], []).append(d)
    keys = list(by); bs = []; sl = []
    for _ in range(B):
        rows = [x for k in RNG.choice(keys, len(keys)) for x in by[k]]
        a, s_ = est(rows)
        if a == a:
            bs.append(a)
        if s_ == s_:
            sl.append(s_)
    return dict(c50=e, lo=float(np.percentile(bs, 2.5)) if bs else float("nan"), hi=float(np.percentile(bs, 97.5)) if bs else float("nan"),
                slope=slope, slope_lo=float(np.percentile(sl, 2.5)) if sl else float("nan"), slope_hi=float(np.percentile(sl, 97.5)) if sl else float("nan"),
                n_boot_valid=len(bs))
