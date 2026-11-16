"""Fill report/report.html placeholders from results/summary.json + report/sections/*.html, then print to PDF with Edge."""
import json, math, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "report")
S = json.load(open(os.path.join(ROOT, "results", "summary.json")))
P = json.load(open(os.path.join(ROOT, "results", "pooled.json")))
ORDER = [m for m in ["gpt-oss-120b", "gpt-oss-20b", "qwen3.8-27b", "allam-2-7b"] if m in S]


def pc(x, d=0):
    return "–" if x is None or x != x else f"{x*100:.{d}f}%"


def ci(t):
    if not t or any(v != v for v in t):
        return "–"
    e, lo, hi = t
    return f"{e*100:.0f}<span class='ci'> [{lo*100:.0f}, {hi*100:.0f}]</span>"


def table1():
    h = ("<table><tr><th>Model</th><th>N</th><th>Accuracy T0</th><th>Mean conf. (right / wrong)</th><th>Wrong answers stated ≥90%</th>"
         "<th>ECE</th><th>AUROC<br>conf→correct</th><th>Re-ask changes<br>right answers</th><th>Re-ask changes<br>wrong answers</th><th>Re-ask fixes<br>wrong answers</th></tr>")
    for m in ORDER:
        s = S[m]
        h += (f"<tr{_row_cls(m)}><td>{m}</td><td>{s['n_valid']}</td><td>{ci(s['acc0'])}</td><td>{pc(s['mean_conf_right'])} / {pc(s['mean_conf_wrong'])}</td>"
              f"<td>{pc(s['share_wrong_conf_ge90'])}</td><td>{s['ece0']:.2f}</td><td>{s['auroc_conf_correct']:.2f}</td><td>{ci(s['neutral_change_right'])}</td><td>{ci(s['neutral_change_wrong'])}</td><td>{ci(s['neutral_fix_wrong'])}</td></tr>")
    return h + "</table>"


def table2():
    h = ("<table><tr><th>Model</th><th>Capitulation<br>C2W</th><th>Correction<br>W2C</th><th>Control<br>W2W</th>"
         "<th>Discernment<br>W2C−C2W</th><th>Latent<br>W2C−W2W</th><th>Defense gap</th><th>AUROC<br>conf→resist</th>"
         "<th>β target-correct<br>(logit)</th><th>r*<br>(consist.)</th></tr>")
    for m in ORDER:
        s = S[m]
        lg = s.get("logit")
        if lg and lg.get("lo"):
            b = f"{lg['coef'][2]:.2f}<span class='ci'> [{lg['lo'][2]:.2f}, {lg['hi'][2]:.2f}]</span>"
        else:
            b = "–"
        au = s.get("auroc_conf_resist_turnlevel", float("nan"))
        auci = s.get("auroc_conf_resist_turnlevel_ci", [float("nan")] * 2)
        aus = "–" if au != au else f"{au:.2f}<span class='ci'> [{auci[0]:.2f}, {auci[1]:.2f}]</span>"
        h += (f"<tr><td>{m}</td><td>{ci(s['C2W_adopt3'])}<span class='ci'> n={s['C2W_n']}</span></td>"
              f"<td>{ci(s['W2C_adopt3'])}<span class='ci'> n={s['W2C_n']}</span></td><td>{ci(s['W2W_adopt3'])}</td>"
              f"<td>{ci(s['discernment3'])}</td><td>{ci(s['latent3'])}</td><td>{ci(s.get('defense_gap'))}</td>"
              f"<td>{aus}</td><td>{b}</td><td>{s.get('best_r', float('nan')):.2f} ({s.get('best_consistency', float('nan')):.2f})</td></tr>")
    return h + "</table>"


def table3():
    h = "<table><tr><th>Model</th><th>C2W, conf&lt;90</th><th>C2W, conf≥90</th><th>W2C, conf&lt;90</th><th>W2C, conf≥90</th><th>W2W, conf&lt;90</th><th>W2W, conf≥90</th></tr>"
    for m in ORDER:
        st = S[m]["strat"]
        cell = lambda k: "–" if st[k][1] == 0 else f"{st[k][0]*100:.0f}%<span class='ci'> ({st[k][1]})</span>"
        h += f"<tr><td>{m}</td>" + "".join(f"<td>{cell(k)}</td>" for k in ["C2W_lo", "C2W_hi", "W2C_lo", "W2C_hi", "W2W_lo", "W2W_hi"]) + "</tr>"
    return h + "</table>"


def examples():
    import html as H
    recs = {}
    for m in ORDER:
        p = os.path.join(ROOT, "results", "raw", f"{m}.jsonl")
        recs[m] = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    qs = {json.loads(l)["qid"]: json.loads(l) for l in open(os.path.join(ROOT, "data", "questions.jsonl"), encoding="utf-8")}
    picks = []
    def find(m, arm, cond):
        for r in recs.get(m, []):
            a = r["arms"].get(arm)
            if a and r["c0"] is not None and cond(r, a):
                return r
    r = find("allam-2-7b", "C2W", lambda r, a: a["turns"][0]["answer"] == a["target"] and r["c0"] >= 80)
    if r: picks.append(("ALLaM-2-7B capitulates at the first doubt", r, "C2W"))
    r = find("gpt-oss-120b", "W2W", lambda r, a: any(t["answer"] == a["target"] for t in a["turns"]))
    if r: picks.append(("gpt-oss-120b, wrong at T0, adopts a different wrong answer (W2W)", r, "W2W"))
    r = find("gpt-oss-120b", "C2W", lambda r, a: len(a["turns"]) == 3 and a["turns"][-1]["answer"] == r["a0"] and r["c0"] >= 90)
    if r: picks.append(("gpt-oss-120b defends a correct answer through all three turns", r, "C2W"))
    out = ""
    for title, r, arm in picks:
        q = qs[r["qid"]]; a = r["arms"][arm]
        tag = {"C2W": "C2W", "W2W": "W2W"}[arm]
        calls = [c for c in r["calls"] if c["tag"] in ("T0", "T0_fmt") or c["tag"].startswith(arm + "_")]
        lines = [f"Q ({q['category']}, gold={r['gold']}): {q['question'][:220]}{'...' if len(q['question'])>220 else ''}"]
        for c in calls:
            lab = c["tag"].replace(arm + "_", "")
            txt = " ".join(c["text"].split())
            if len(txt) > 260: txt = "..." + txt[-260:]
            if lab.startswith("P"):
                lines.append(f"[user {lab}] pushes ({a['target']})")
            lines.append(f"[model {lab}] {txt}")
        out += f"<h3>{H.escape(title)}</h3><pre>" + H.escape(chr(10).join(lines)) + "</pre>"
    return out


def _ci3(t, pp=False):
    if not t or any(v != v for v in t):
        return "–"
    z = lambda q: ("0" if f"{q*100:.0f}" == "-0" else f"{q*100:.0f}")
    return f"{z(t[0])}<span class='ci'> [{z(t[1])}, {z(t[2])}]</span>"


def _pv(p):
    if p != p:
        return "–"
    return "&lt;.001" if p < 0.001 else (f"{p:.3f}".lstrip("0") if p < 0.995 else "1.00")


def _row_cls(m):
    return " class='dim'" if m == "qwen3.8-27b" else ""


def table_adopt():
    h = ("<table><tr><th>Model</th><th>Capitulation<br>(C2W)</th><th>Correction<br>(W2C)</th><th>Control<br>(W2W)</th>"
         "<th>Discernment<br>W2C − C2W</th><th>Fisher<br><i>p</i></th><th>Latent discernment<br>W2C − W2W (paired)</th><th>McNemar<br><i>p</i></th>"
         "<th>W2C beyond<br>neutral re-ask</th></tr>")
    for m in ORDER + ["gptoss-pooled"]:
        if m not in S:
            continue
        x = S[m]["x"]
        if m == "gptoss-pooled":
            h += (f"<tr class='pooled'><td>gpt-oss pooled</td><td colspan='3' style='text-align:center'>(both gpt-oss models)</td><td>–</td><td>{_pv(x['fisher_w2c_vs_c2w']['p'])}</td>"
                  f"<td>{_ci3(x['latent_paired'])}<span class='ci'> n={x['mcnemar']['n_pairs']}</span></td><td>{_pv(x['mcnemar']['p'])}</td><td>{_ci3(x['w2c_minus_neutral_fix'])}</td></tr>")
            continue
        s_ = S[m]
        h += (f"<tr{_row_cls(m)}><td>{m}</td><td>{_ci3(s_['C2W_adopt3'])}<span class='ci'> n={s_['C2W_n']}</span></td>"
              f"<td>{_ci3(s_['W2C_adopt3'])}<span class='ci'> n={s_['W2C_n']}</span></td><td>{_ci3(s_['W2W_adopt3'])}</td>"
              f"<td>{_ci3(s_['discernment3'])}</td><td>{_pv(x['fisher_w2c_vs_c2w']['p'])}</td>"
              f"<td>{_ci3(x['latent_paired'])}</td><td>{_pv(x['mcnemar']['p'])}</td><td>{_ci3(x['w2c_minus_neutral_fix'])}</td></tr>")
    return h + "</table>"


def table_conf():
    h = ("<table><tr><th>Model</th><th>AUROC conf→resist<br>turn 1 only</th><th>AUROC<br>wrong-T0 arms</th><th>AUROC<br>all turns</th>"
         "<th>OR per +10 pts<br>confidence</th><th>Threshold <i>ĉ</i><sub>50</sub><br>(= implied <i>r</i>)</th><th>Defense gap<br>(C2W)</th></tr>")
    f2 = lambda t: "–" if (not t or t[0] != t[0]) else f"{t[0]:.2f}<span class='ci'> [{t[1]:.2f}, {t[2]:.2f}]</span>"
    for m in ORDER + ["gptoss-pooled"]:
        if m not in S:
            continue
        x = S[m]["x"]; c = x.get("c50") or {}
        if c and c.get("slope") == c.get("slope") and c.get("slope", 0) < 0 and c.get("c50") == c.get("c50"):
            orr = f"{math.exp(c['slope']):.2f}<span class='ci'> [{math.exp(c['slope_lo']):.2f}, {math.exp(c['slope_hi']):.2f}]</span>"
            cc = f"{c['c50']:.2f}<span class='ci'> [{c['lo']:.2f}, {c['hi']:.2f}]</span>"
        else:
            orr = (f"{math.exp(c['slope']):.2f}<span class='ci'> [{math.exp(c['slope_lo']):.2f}, {math.exp(c['slope_hi']):.2f}]</span>" if c and c.get('slope') == c.get('slope') else "–")
            cc = "none (ceiling)"
        name = "gpt-oss pooled" if m == "gptoss-pooled" else m
        gap = _ci3(S[m].get("defense_gap")) if m in ORDER else "–"
        cls = " class='pooled'" if m == "gptoss-pooled" else _row_cls(m)
        h += (f"<tr{cls}><td>{name}</td><td>{f2(x['auroc_turn1'])}</td><td>{f2(x['auroc_wrong'])}</td><td>{f2(x['auroc_all_clustered'])}</td>"
              f"<td>{orr}</td><td>{cc}</td><td>{gap}</td></tr>")
    return h + "</table>"


def table_strat():
    h = ("<table><tr><th>Model</th><th>C2W<br>conf&lt;90</th><th>C2W<br>conf≥90</th><th>W2C<br>conf&lt;90</th><th>W2C<br>conf≥90</th>"
         "<th>W2W<br>conf&lt;90</th><th>W2W<br>conf≥90</th><th>Keep rate (C2W) by T0 confidence<br>&lt;90 / 90–94 / 95–100</th></tr>")
    for m in ORDER + ["gptoss-pooled"]:
        if m not in S:
            continue
        if m == "gptoss-pooled":
            kb = S[m]["x"]["keep_bins"]
            kk = " / ".join("–" if kb[k]['n'] == 0 else f"{kb[k]['keep']*100:.0f}%<span class='ci'> ({kb[k]['n']})</span>" for k in ["<90", "90-94", "95-100"])
            h += f"<tr class='pooled'><td>gpt-oss pooled</td><td colspan='6'></td><td>{kk}</td></tr>"
            continue
        st = S[m]["strat"]; kb = S[m]["x"]["keep_bins"]
        cell = lambda k: "–" if st[k][1] == 0 else f"{st[k][0]*100:.0f}%<span class='ci'> ({st[k][1]})</span>"
        kk = " / ".join("–" if kb[k]['n'] == 0 else f"{kb[k]['keep']*100:.0f}%<span class='ci'> ({kb[k]['n']})</span>" for k in ["<90", "90-94", "95-100"])
        h += f"<tr{_row_cls(m)}><td>{m}</td>" + "".join(f"<td>{cell(k)}</td>" for k in ["C2W_lo", "C2W_hi", "W2C_lo", "W2C_hi", "W2W_lo", "W2W_hi"]) + f"<td>{kk}</td></tr>"
    return h + "</table>"


def main():
    html = open(os.path.join(REP, "report.html"), encoding="utf-8").read()
    secdir = os.path.join(REP, "sections")
    for f in sorted(os.listdir(secdir)) if os.path.isdir(secdir) else []:
        key = os.path.splitext(f)[0].upper()
        html = html.replace(f"[[{key}]]", open(os.path.join(secdir, f), encoding="utf-8").read())
    def val(m, key):
        v = S[m]
        for k in key.split("."):
            v = v[int(k)] if isinstance(v, list) else v[k]
        return v

    def sub(mo):
        kind, m, key = mo.group(1), mo.group(2), mo.group(3)
        v = val(m, key)
        if kind == "P":   # percent of scalar or of first elem of triple
            x = v[0] if isinstance(v, list) else v
            return f"{x*100:.0f}%"
        if kind == "PCI":
            return f"{v[0]*100:.0f}% [{v[1]*100:.0f}, {v[2]*100:.0f}]"
        if kind == "PP":  # percentage points
            x = v[0] if isinstance(v, list) else v
            o = f"{x*100:.0f}"; return ("0" if o == "-0" else o) + " pp"
        if kind == "PPCI":
            z = lambda q: ("0" if f"{q*100:.0f}" == "-0" else f"{q*100:.0f}")
            return f"{z(v[0])} pp [{z(v[1])}, {z(v[2])}]"
        if kind == "F":
            x = v[0] if isinstance(v, list) else v
            out = f"{x:.2f}"
            return "0.00" if out == "-0.00" else out
        if kind == "PV":
            x = v["p"] if isinstance(v, dict) else v
            return "<i>p</i> &lt; .001" if x < 0.001 else ("<i>p</i> = " + f"{x:.3f}".lstrip("0") if x < 0.995 else "<i>p</i> = 1.00")
        if kind == "OR":
            return f"{math.exp(v):.2f}"
        if kind == "ORRED":
            return f"{(1 - math.exp(v)) * 100:.0f}%"
        if kind == "N":
            return str(v)
        return mo.group(0)
    html = re.sub(r"\[\[(PCI|PPCI|PP|PV|ORRED|OR|P|F|N):([a-z0-9.\-]+):([A-Za-z0-9_.]+)\]\]", sub, html)
    nitems = sum(S[m]["n_valid"] for m in ORDER)
    html = html.replace("[[TABLE1]]", table1()).replace("[[TABLE2]]", table_adopt()).replace("[[TABLE2B]]", table_conf()).replace("[[TABLE3]]", table_strat()).replace("[[EXAMPLES]]", examples())
    html = html.replace("[[NITEMS]]", f"{nitems:,}").replace("[[NCALLS]]", f"{P['n_calls']:,}")
    html = html.replace("[[NQ]]", "201").replace("[[NTOKENS]]", f"{P['tokens']/1e6:.1f}M")
    html = html.replace("[[PARSEFAIL]]", f"{P['parse_fail_calls']}")
    logtxt = open(os.path.join(ROOT, "results", "run.log"), encoding="utf-8").read()
    fails = {m: len(set(re.findall(r"\[" + re.escape(m) + r"\] q(\d+) failed", logtxt))) for m in ORDER}
    html = html.replace("[[APIFAIL]]", ", ".join(f"{m}: {n}" for m, n in fails.items()))
    html = html.replace("[[APIFAILTOTAL]]", str(sum(fails.values())))
    html = html.replace("</style>", "tr.dim td{color:#777} tr.pooled td{border-top:0.5px solid #999;font-style:italic} </style>")
    left = re.findall(r"\[\[[A-Z0-9_]+\]\]", html)
    if left:
        print("UNFILLED:", sorted(set(left)))
    out_html = os.path.join(REP, "Confidence_Under_Pushback_Report.html")
    open(out_html, "w", encoding="utf-8").write(html)
    edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    pdf = os.path.join(REP, "Confidence_Under_Pushback_Report.pdf")
    url = "file:///" + out_html.replace("\\", "/")
    subprocess.run([edge, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", url],
                   capture_output=True, timeout=120)
    try:
        import pymupdf
        d = pymupdf.open(pdf)
        print("PDF pages:", len(d))
        # find page where References starts (8-page cap excludes refs+appendix)
        for i, pg in enumerate(d):
            if "References" in pg.get_text():
                print("References heading on page", i + 1)
                break
        if len(sys.argv) > 1:
            for i, pg in enumerate(d):
                pg.get_pixmap(dpi=int(sys.argv[2]) if len(sys.argv) > 2 else 60).save(os.path.join(sys.argv[1], f"p{i+1}.png"))
    except Exception as e:
        print("preview skipped:", e)
    # slides: same placeholder substitution, landscape PDF
    sp = os.path.join(REP, "slides.html")
    if os.path.exists(sp):
        sl = open(sp, encoding="utf-8").read()
        sl = re.sub(r"\[\[(PCI|PPCI|PP|PV|ORRED|OR|P|F|N):([a-z0-9.\-]+):([A-Za-z0-9_.]+)\]\]", sub, sl)
        sl = sl.replace("[[NITEMS]]", f"{nitems:,}").replace("[[NCALLS]]", f"{P['n_calls']:,}").replace("[[NQ]]", "201")
        left = re.findall(r"\[\[[^\]]+\]\]", sl)
        if left:
            print("SLIDES UNFILLED:", sorted(set(left)))
        so = os.path.join(REP, "_slides_rendered.html")
        open(so, "w", encoding="utf-8").write(sl)
        spdf = os.path.join(REP, "Confidence_Under_Pushback_Slides.pdf")
        subprocess.run([edge, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={spdf}",
                        "file:///" + so.replace("\\", "/")], capture_output=True, timeout=120)
        try:
            import pymupdf
            d = pymupdf.open(spdf); print("Slides pages:", len(d))
            if len(sys.argv) > 1:
                for i, pg in enumerate(d):
                    pg.get_pixmap(dpi=40).save(os.path.join(sys.argv[1], f"s{i+1}.png"))
        except Exception as e:
            print("slides preview skipped:", e)


if __name__ == "__main__":
    main()
