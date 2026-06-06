"""
score_eval.py — попарное сравнение результатов разных архитектур.

Считает:
  • Сводную таблицу метрик по каждой архитектуре
  • Разбивку по сценариям A-K
  • Список кейсов, где архитектуры расходятся
  • Критерий Вилкоксона для попарного сравнения (требуется в 2.5.5)

Использование:
    python score_eval.py eval_results_arch1.jsonl eval_results_arch2.jsonl [...]
"""

import json
import sys
from collections import defaultdict
from statistics import median, mean

try:
    from scipy.stats import wilcoxon
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def stats(results):
    total = len(results)
    passed = sum(1 for r in results if r["auto_passed"])
    problem = [r for r in results if r["problem_case"]]
    prob_ok = sum(1 for r in problem if r["auto_passed"])
    times = [r.get("elapsed_sec", 0) for r in results]
    sorted_t = sorted(times)
    p95 = sorted_t[int(len(sorted_t) * 0.95)] if len(sorted_t) > 1 else (sorted_t[0] if sorted_t else 0)
    excs = sum(1 for r in results if r.get("exception"))

    # Retrieval — только по кейсам с GT
    ret = [r["retrieval_metrics"] for r in results if r.get("retrieval_metrics")]
    retr_avg = {}
    if ret:
        for k in ["P@3", "P@5", "R@3", "R@5", "MRR", "nDCG@5"]:
            retr_avg[k] = mean(m[k] for m in ret)

    by_scenario = defaultdict(lambda: {"total": 0, "passed": 0, "r5": [], "mrr": []})
    for r in results:
        s = r["scenario"]
        by_scenario[s]["total"] += 1
        if r["auto_passed"]:
            by_scenario[s]["passed"] += 1
        if r.get("retrieval_metrics"):
            by_scenario[s]["r5"].append(r["retrieval_metrics"]["R@5"])
            by_scenario[s]["mrr"].append(r["retrieval_metrics"]["MRR"])

    return {
        "total":          total,
        "passed":         passed,
        "pct":            100 * passed / total if total else 0,
        "problem_total":  len(problem),
        "problem_passed": prob_ok,
        "problem_pct":    100 * prob_ok / len(problem) if problem else 0,
        "exceptions":     excs,
        "median_sec":     median(times) if times else 0,
        "p95_sec":        p95,
        "mean_sec":       mean(times) if times else 0,
        "retrieval":      retr_avg,
        "by_scenario":    dict(by_scenario),
    }


def print_comparison(files):
    all_data = {}
    for path in files:
        data = load(path)
        arch = data[0]["arch"] if data else path
        all_data[arch] = (stats(data), data)

    archs = list(all_data.keys())

    # ─── Сводная таблица ───────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("СВОДНАЯ ТАБЛИЦА")
    print(f"{'='*100}")
    print(f"{'Метрика':<28} " + "  ".join(f"{a:<22}" for a in archs))
    print("-" * (28 + 24 * len(archs)))

    rows = [
        ("Всего кейсов",         lambda s: str(s["total"])),
        ("Auto-pass",            lambda s: f"{s['passed']}/{s['total']} ({s['pct']:.1f}%)"),
        ("Проблемные пройдены",  lambda s: f"{s['problem_passed']}/{s['problem_total']} ({s['problem_pct']:.1f}%)"),
        ("Исключения",           lambda s: str(s["exceptions"])),
        ("",                     lambda s: ""),
        ("--- Retrieval ---",    lambda s: ""),
        ("P@3",                  lambda s: f"{s['retrieval'].get('P@3', 0):.4f}"),
        ("P@5",                  lambda s: f"{s['retrieval'].get('P@5', 0):.4f}"),
        ("R@3",                  lambda s: f"{s['retrieval'].get('R@3', 0):.4f}"),
        ("R@5",                  lambda s: f"{s['retrieval'].get('R@5', 0):.4f}"),
        ("MRR",                  lambda s: f"{s['retrieval'].get('MRR', 0):.4f}"),
        ("nDCG@5",               lambda s: f"{s['retrieval'].get('nDCG@5', 0):.4f}"),
        ("",                     lambda s: ""),
        ("--- Латентность ---",  lambda s: ""),
        ("Median (сек)",         lambda s: f"{s['median_sec']:.2f}"),
        ("p95 (сек)",            lambda s: f"{s['p95_sec']:.2f}"),
        ("Mean (сек)",           lambda s: f"{s['mean_sec']:.2f}"),
    ]
    for label, fn in rows:
        print(f"{label:<28} " + "  ".join(f"{fn(all_data[a][0]):<22}" for a in archs))

    # ─── По сценариям ──────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("ПО СЦЕНАРИЯМ")
    print(f"{'='*100}")
    print(f"{'Сценарий':<12} " + "  ".join(f"{a:<28}" for a in archs))
    print("-" * (12 + 30 * len(archs)))

    all_sc = sorted(set(s for a in archs for s in all_data[a][0]["by_scenario"]))
    for sc in all_sc:
        row = []
        for a in archs:
            d = all_data[a][0]["by_scenario"].get(sc, {"total": 0, "passed": 0, "r5": [], "mrr": []})
            pct = 100 * d["passed"] / d["total"] if d["total"] else 0
            base = f"{d['passed']}/{d['total']} ({pct:>4.0f}%)"
            if d["r5"]:
                base += f" R5={mean(d['r5']):.2f}"
            row.append(f"{base:<28}")
        print(f"{sc:<12} " + "  ".join(row))

    # ─── Расхождения ────────────────────────────────────────────────────────
    if len(archs) >= 2:
        print(f"\n{'='*100}")
        print("РАСХОЖДЕНИЯ (кейс прошёл хотя бы в одной, не прошёл хотя бы в одной)")
        print(f"{'='*100}")

        all_ids = sorted(set(r["id"] for a in archs for r in all_data[a][1]))
        diff_count = 0
        for cid in all_ids:
            by_arch = {a: next((r for r in all_data[a][1] if r["id"] == cid), None) for a in archs}
            statuses = {a: (r["auto_passed"] if r else None) for a, r in by_arch.items()}
            vals = [v for v in statuses.values() if v is not None]
            if len(set(vals)) > 1:  # есть и True и False
                diff_count += 1
                meta = next(r for r in by_arch.values() if r)
                prob = "(!)" if meta.get("problem_case") else "   "
                line = f"  {prob} {cid:<7} | " + " | ".join(f"{a}: {'v' if statuses[a] else 'x'}" for a in archs)
                desc = meta.get("scenario_desc", "")[:50]
                print(f"{line:<70} {desc}")
        if diff_count == 0:
            print("  Все архитектуры дают одинаковые результаты на auto-check.")
        else:
            print(f"\n  Всего расхождений: {diff_count}")

    # ─── Критерий Вилкоксона ────────────────────────────────────────────────
    if HAS_SCIPY and len(archs) == 2:
        print(f"\n{'='*100}")
        print(f"КРИТЕРИЙ ВИЛКОКСОНА (попарное сравнение, alpha=0.05)")
        print(f"{'='*100}")
        a1, a2 = archs
        ids = sorted(set(r["id"] for r in all_data[a1][1]) & set(r["id"] for r in all_data[a2][1]))

        def get_metric(arch, cid, key, default=0):
            r = next((x for x in all_data[arch][1] if x["id"] == cid), None)
            if not r:
                return default
            if key == "auto_passed":
                return 1 if r["auto_passed"] else 0
            if key == "elapsed_sec":
                return r.get("elapsed_sec", 0)
            return r.get("retrieval_metrics", {}).get(key, default)

        for metric in ["auto_passed", "P@5", "R@5", "MRR", "nDCG@5", "elapsed_sec"]:
            v1 = [get_metric(a1, cid, metric) for cid in ids]
            v2 = [get_metric(a2, cid, metric) for cid in ids]
            # Wilcoxon требует ненулевые разности
            diffs = [a - b for a, b in zip(v1, v2)]
            if all(d == 0 for d in diffs):
                print(f"  {metric:<14}: разностей нет (метрики идентичны)")
                continue
            try:
                stat, p = wilcoxon(v1, v2, zero_method="wilcox")
                sig = "ЗНАЧИМО" if p < 0.05 else "не значимо"
                print(f"  {metric:<14}: W={stat:.2f}  p={p:.4f}  [{sig}]  "
                      f"mean({a1})={mean(v1):.3f}  mean({a2})={mean(v2):.3f}")
            except Exception as e:
                print(f"  {metric:<14}: не удалось рассчитать ({e})")
    elif not HAS_SCIPY:
        print(f"\n[INFO] Установи scipy для расчёта критерия Вилкоксона: pip install scipy")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python score_eval.py results1.jsonl [results2.jsonl ...]")
        sys.exit(1)
    print_comparison(sys.argv[1:])
