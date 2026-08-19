"""
Jinpeng Zhai 914962409@qq.com
Analyze the judge-validation runs (SKAG workshop revision).

Two analyses, both text-only (no images):

  stability   Repeat-run label-flip rate for the Qwen 3.6 judge over the
              200-sample subset (Q1-Q3). Answers: "does sampling noise flip
              safe/unsafe labels, or are the reported values stable?"
  crossjudge  Qwen 3.6 (Q0, full row) vs Gemma-4 (G0, full row) on the SAME
              samples/types. Answers: "are the results an artifact of the
              Qwen judge (circularity), or do they hold under an independent
              judge family?"

Output: markdown tables to stdout + a JSON blob to --out (optional).

Run:
  $PY rag/analyze_judge.py stability
  $PY rag/analyze_judge.py crossjudge
  $PY rag/analyze_judge.py all
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "result")
VLM = "internvl3_5-8b"
TYPES = ("original_response", "rag_on_response")  # baseline, ours
TNAME = {"original_response": "baseline", "rag_on_response": "ours"}
BENCHES = ("SIUO", "MSSB", "MMSB", "SPAVLH")

STABILITY_RUNS = ("run1", "run2", "run3")   # Qwen 3.6, 200-subset
CROSS_QWEN = ""                             # Q0: JUDGED_<b>_<vlm>.json (full row)
CROSS_GEMMA = "_gemma4_full"                # G0: JUDGED_<b>_<vlm>_gemma4_full.json


def _load(bench, out_suffix=""):
    p = os.path.join(RES, "JUDGED_%s_%s%s.json" % (bench, VLM, out_suffix))
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return {s["image"]: s for s in json.load(f)}


def _labels(rec, t):
    """is_safe for response type t, or None if not judged."""
    if rec is None or t not in rec:
        return None
    v = rec[t].get("is_safe")
    return v if isinstance(v, bool) else None


def safe_rate(labels):
    vals = [x for x in labels if isinstance(x, bool)]
    if not vals:
        return None, 0
    return sum(vals) / len(vals), len(vals)


# ---------------------------------------------------------------- stability
def stability():
    rows = []
    for bench in BENCHES:
        files = [_load(bench, "_qwen36_%s" % r) for r in STABILITY_RUNS]
        if any(f is None for f in files):
            rows.append({"bench": bench, "status": "MISSING FILES"})
            continue
        imgs = set(files[0]) & set(files[1]) & set(files[2])
        for t in TYPES:
            # collect per-run label lists, keyed by image
            L = {r: {} for r in STABILITY_RUNS}
            for img in imgs:
                for r in STABILITY_RUNS:
                    L[r][img] = _labels(files[STABILITY_RUNS.index(r)][img], t)
            # pairwise flip counts
            def pair(a, b):
                n = fl = 0
                for img in imgs:
                    la, lb = L[a][img], L[b][img]
                    if isinstance(la, bool) and isinstance(lb, bool):
                        n += 1
                        if la != lb:
                            fl += 1
                return fl, n
            f12, n12 = pair("run1", "run2")
            f13, n13 = pair("run1", "run3")
            f23, n23 = pair("run2", "run3")
            # per-sample: does any run disagree with the other two (a flip)?
            n_flip = n_agree = 0
            per_run_rates = {r: [] for r in STABILITY_RUNS}
            for img in imgs:
                vals = [L[r][img] for r in STABILITY_RUNS]
                for r in STABILITY_RUNS:
                    if isinstance(vals[STABILITY_RUNS.index(r)], bool):
                        per_run_rates[r].append(vals[STABILITY_RUNS.index(r)])
                if all(isinstance(v, bool) for v in vals):
                    if vals[0] == vals[1] == vals[2]:
                        n_agree += 1
                    else:
                        n_flip += 1
            rates = [safe_rate(per_run_rates[r])[0] for r in STABILITY_RUNS]
            rows.append({
                "bench": bench, "type": TNAME[t],
                "n_samples": len(imgs),
                "rate_baseline" if t == "original_response" else "rate_ours":
                    None if all(x is None for x in rates) else rates,
                "pairwise_flips": {"r1-r2": [f12, n12], "r1-r3": [f13, n13], "r2-r3": [f23, n23]},
                "any_flip": n_flip, "all_agree": n_agree,
                "flip_rate": (n_flip / (n_flip + n_agree)) if (n_flip + n_agree) else None,
            })
    return rows


def _fmt_rate_list(rs):
    if not rs or all(r is None for r in rs):
        return "n/a"
    return " / ".join(("-" if r is None else "%.1f%%" % (100 * r)) for r in rs)


def stability_md(rows):
    out = ["## Judge stability (Qwen 3.6, 200-subset, 3 repeat runs)\n"]
    out.append("| bench | type | samples | safe% r1/r2/r3 | flips r1-2 r1-3 r2-3 | any-flip | any-flip rate |")
    out.append("|---|---|---|---|---|---|---|")
    tot_flip = tot_n = 0
    for r in rows:
        if "status" in r:
            out.append("| %s | - | - | - | - | - | %s |" % (r["bench"], r["status"]))
            continue
        pf = r["pairwise_flips"]
        flipcol = "%d/%d %d/%d %d/%d" % (
            pf["r1-r2"][0], pf["r1-r2"][1], pf["r1-r3"][0], pf["r1-r3"][1], pf["r2-r3"][0], pf["r2-r3"][1])
        rate = r.get("rate_baseline", r.get("rate_ours"))
        fr = "-" if r["flip_rate"] is None else "%.1f%%" % (100 * r["flip_rate"])
        out.append("| %s | %s | %d | %s | %s | %d | %s |" % (
            r["bench"], r["type"], r["n_samples"], _fmt_rate_list(rate), flipcol, r["any_flip"], fr))
        tot_flip += r["any_flip"]; tot_n += r["any_flip"] + r["all_agree"]
    out.append("")
    out.append("**Overall any-flip rate (all benches, both types): %.2f%%  (%d / %d samples flip on at least one run)**"
               % (100 * tot_flip / tot_n if tot_n else 0.0, tot_flip, tot_n))
    return "\n".join(out)


# ---------------------------------------------------------------- crossjudge
def _pct(x, nd=1):
    return "-" if x is None else f"{100*x:.{nd}f}%"


def stability_paper(rows):
    """Compact publication table: mean±std safe rate (3 runs) + label
    reproduction rate, one row per bench. Details stay in the full table."""
    lines = [
        "% Judge stability — Qwen-3.6 judge, 3 independent runs, 200-sample subset",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Judge stability across 3 independent runs (temperature 0.7). "
        "Safe\\% is the per-run safe-response rate (mean $\\pm$ std); "
        "\\textit{reproducible} is the fraction of samples whose safe/unsafe label is "
        "identical in all 3 runs (baseline + ours pooled per benchmark).}",
        "\\label{tab:stability}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Benchmark & Baseline safe\\% (3 runs) & +Ours safe\\% (3 runs) & Labels reproducible\\\\",
        "\\midrule",
    ]
    tot_flip = tot_n = 0
    for bench in BENCHES:
        br = [r for r in rows if r.get("bench") == bench and r["type"] == "baseline"]
        orow = [r for r in rows if r.get("bench") == bench and r["type"] == "ours"]
        if not br or not orow or "status" in br[0]:
            lines.append("%s & - & - & (pending)\\\\" % bench)
            continue
        rb, ro = br[0].get("rate_baseline"), orow[0].get("rate_ours")
        def ms(rs):
            vals = [v for v in rs if isinstance(v, (int, float))]
            if not vals:
                return "-"
            m = sum(vals) / len(vals)
            var = sum((v - m) ** 2 for v in vals) / max(len(vals) - 1, 1)
            return "%.1f \\,${\\pm}\\,$\\,%.1f" % (100 * m, 100 * var ** 0.5)
        n_flip = br[0]["any_flip"] + orow[0]["any_flip"]
        n_all = br[0]["any_flip"] + br[0]["all_agree"] + orow[0]["any_flip"] + orow[0]["all_agree"]
        tot_flip += n_flip; tot_n += n_all
        repro = f"{100*(1 - n_flip/n_all):.1f}%" if n_all else "-"
        lines.append("%s & %s & %s & %s\\\\" % (bench, ms(rb), ms(ro), repro))
    repro_all = f"{100*(1 - tot_flip/tot_n):.1f}%" if tot_n else "-"
    lines += [
        "\\midrule",
        "\\textbf{Overall} & & & \\textbf{%s}\\\\" % repro_all,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def crossjudge_paper(rows):
    """Compact publication table: ours-baseline delta per judge + agreement."""
    lines = [
        "% Cross-judge validation — Qwen-3.6 (Q0) vs Gemma-4 (G0), full InternVL row",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Cross-judge validation on the full InternVL-3.5-8B row. "
        "$\\Delta$ is the safe-response-rate gain of +Ours over baseline as measured by each "
        "independent judge; agreement is the fraction of samples on which the two judges "
        "concur in the safe/unsafe label (baseline + ours pooled).}",
        "\\label{tab:crossjudge}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Benchmark & $\\Delta$ (Qwen-3.6) & $\\Delta$ (Gemma-4) & Label agreement\\\\",
        "\\midrule",
    ]
    for bench in BENCHES:
        br = [r for r in rows if r.get("bench") == bench and r["type"] == "baseline"]
        orow = [r for r in rows if r.get("bench") == bench and r["type"] == "ours"]
        if not br or not orow or "status" in br[0]:
            lines.append("%s & - & - & (pending)\\\\" % bench)
            continue
        def rate(rs, key):
            n = sum(r["n"] for r in rs)
            if not n:
                return None
            return sum(r[key] * r["n"] for r in rs if r[key] is not None) / n
        qb, qo = rate(br, "qwen_rate"), rate(orow, "qwen_rate")
        gb, go = rate(br, "gemma_rate"), rate(orow, "gemma_rate")
        n = sum(r["n"] for r in br + orow)
        agr = (sum(r["agree"] for r in br + orow) / n) if n else None
        def d(a, b):
            if a is None or b is None:
                return "-"
            return f"+{100*(a-b):.1f}"
        lines.append("%s & %s & %s & %s\\\\" % (
            bench, d(qo, qb), d(go, gb), _pct(agr)))
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- crossjudge-data
def crossjudge():
    rows = []
    for bench in BENCHES:
        q = _load(bench, CROSS_QWEN)
        g = _load(bench, CROSS_GEMMA)
        if q is None or g is None:
            rows.append({"bench": bench, "status": "MISSING (Q0 or G0 not present)"})
            continue
        imgs = set(q) & set(g)
        for t in TYPES:
            ql = [(_labels(q[i], t), _labels(g[i], t)) for i in imgs]
            both = [(a, b) for a, b in ql if isinstance(a, bool) and isinstance(b, bool)]
            agree = sum(1 for a, b in both if a == b)
            qr, nq = safe_rate([a for a, _ in both])
            gr, ng = safe_rate([b for _, b in both])
            # per-run (per-judge) safe rate over ALL judged (not just both-present)
            rows.append({
                "bench": bench, "type": TNAME[t], "n": len(both),
                "qwen_rate": qr, "gemma_rate": gr,
                "agree": agree, "agree_rate": agree / len(both) if both else None,
            })
    return rows


def crossjudge_md(rows):
    out = ["## Cross-judge (Qwen 3.6 Q0  vs  Gemma-4 G0, full row, same samples)\n"]
    out.append("| bench | type | n | Qwen safe% | Gemma safe% | label agreement |")
    out.append("|---|---|---|---|---|---|")
    for r in rows:
        if "status" in r:
            out.append("| %s | - | - | - | - | %s |" % (r["bench"], r["status"]))
            continue
        ar = "-" if r["agree_rate"] is None else "%.1f%%" % (100 * r["agree_rate"])
        out.append("| %s | %s | %d | %s | %s | %s |" % (
            r["bench"], r["type"], r["n"],
            "-" if r["qwen_rate"] is None else "%.1f%%" % (100 * r["qwen_rate"]),
            "-" if r["gemma_rate"] is None else "%.1f%%" % (100 * r["gemma_rate"]),
            ar))
    out.append("")
    out.append("safe% = fraction the judge labels safe. agreement = fraction of samples where both judges concur on the safe/unsafe label.")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["stability", "crossjudge", "all", "paper"], nargs="?", default="all")
    ap.add_argument("--out", default="", help="optional path to dump the raw rows as JSON")
    a = ap.parse_args()

    result = {}
    if a.what in ("stability", "all", "paper"):
        s = stability()
        result["stability"] = s
        if a.what == "paper":
            print(stability_paper(s)); print()
        else:
            print(stability_md(s)); print()
    if a.what in ("crossjudge", "all", "paper"):
        c = crossjudge()
        result["crossjudge"] = c
        if a.what == "paper":
            print(crossjudge_paper(c)); print()
        else:
            print(crossjudge_md(c)); print()

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print("(rows written to %s)" % a.out)


if __name__ == "__main__":
    main()
