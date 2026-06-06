"""
validation/run_eval.py — прогон валидационного датасета через бот.

Запускать ИЗ КОРНЯ ПРОЕКТА:
    python validation/run_eval.py --arch baseline
    python validation/run_eval.py --arch hybrid_rerank --limit 5

Собирает все метрики, требуемые в главах 2-3 ВКР:
  • Retrieval:    P@3, P@5, R@3, R@5, MRR, nDCG@5
  • Ответ:        auto_pass (keyword check), %-прохождения
  • Латентность:  median, p95, mean
  • Стоимость:    # LLM-вызовов на запрос
  • Робастность:  exceptions count

Бот должен экспонировать (это уже встроено в base):
  • bot.last_retrieved_programs — список имён программ после поиска
  • bot.last_llm_calls — # LLM-вызовов на последний запрос
  • bot.respond(text) или .process_message(text)
  • bot.reset() — сброс состояния
"""

import json
import time
import argparse
import sys
import os
import math
import traceback
from statistics import median, mean

# ─── Импорт бота ─────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def init_bot():
    """Инициализирует бота с настоящими зависимостями (как app.py)."""
    from config.prompts import MODEL_PRIORITY
    from services.api_key_manager.api_key_manager import SmartKeyManager, Logger, load_api_keys
    from services.rag.rag_chatbot import RAGChatbot

    db_path = os.path.join(PROJECT_ROOT, "services", "db", "knowledge.db")
    stats_file = os.path.join(PROJECT_ROOT, "services", "api_key_manager", "key_stats.json")

    keys = load_api_keys(PROJECT_ROOT)
    if not keys:
        raise RuntimeError("API-ключи не найдены. Проверь services/api_key_manager/api_accounts.xlsx")

    logger = Logger("eval")
    km = SmartKeyManager(keys, logger, cooldown_seconds=60, stats_file=stats_file)
    bot = RAGChatbot(db_path, km, logger, MODEL_PRIORITY)
    return bot


# ─── Каталог программ (fallback-извлечение из текста) ───────────────────────
def load_catalog():
    """Список канонических названий программ."""
    catalog_path = os.path.join(os.path.dirname(__file__), "programs_catalog.json")
    if os.path.exists(catalog_path):
        with open(catalog_path, encoding="utf-8") as f:
            return json.load(f)
    return []


# ─── Вызов бота ───────────────────────────────────────────────────────────────
def call_bot(bot, user_input, catalog):
    """Возвращает (response_text, retrieved_program_names, llm_calls)."""
    if hasattr(bot, "respond"):
        response = bot.respond(user_input)
    else:
        response = bot.process_message(user_input)

    # 1. Предпочитаемый способ — атрибут бота
    retrieved = list(getattr(bot, "last_retrieved_programs", []) or [])

    # 2. Fallback: ищем названия программ в тексте ответа
    if not retrieved and catalog:
        text_lower = (response or "").lower()
        found = []
        for name in catalog:
            key = name
            for q in ['"', '«', '»']:
                if q in key:
                    parts = key.split(q)
                    if len(parts) >= 3:
                        key = parts[1]
                        break
            pos = text_lower.find(key.lower())
            if pos >= 0:
                found.append((name, pos))
        found.sort(key=lambda x: x[1])
        retrieved = [n for n, _ in found]

    llm_calls = getattr(bot, "last_llm_calls", None)
    return response, retrieved, llm_calls


def run_single(bot, user_input, catalog):
    bot.reset()
    return call_bot(bot, user_input, catalog)


def run_multi(bot, turns, catalog):
    bot.reset()
    last_resp, last_ret, total_llm = None, [], 0
    for turn in turns:
        if turn["role"] == "user":
            resp, ret, llm = call_bot(bot, turn["content"], catalog)
            last_resp, last_ret = resp, ret
            if llm is not None:
                total_llm += llm
        # assistant-ходы с __BOT_RESPONSE__ ничего не делают — бот сам ведёт историю
    return last_resp, last_ret, total_llm


# ─── Метрики ─────────────────────────────────────────────────────────────────
def precision_at_k(retrieved, gt, k):
    if not retrieved:
        return 0.0
    top = retrieved[:k]
    return sum(1 for r in top if r in gt) / k


def recall_at_k(retrieved, gt, k):
    if not gt:
        return 0.0
    top = retrieved[:k]
    return sum(1 for r in top if r in gt) / len(gt)


def reciprocal_rank(retrieved, gt):
    for i, r in enumerate(retrieved):
        if r in gt:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved, gt, k):
    if not gt:
        return 0.0
    gt_set = set(gt)
    dcg = sum((1.0 if r in gt_set else 0.0) / math.log2(i + 2)
              for i, r in enumerate(retrieved[:k]))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gt), k)))
    return dcg / idcg if idcg > 0 else 0.0


def auto_check(case, response):
    if response is None:
        return False, [], list(case.get("expected_not_contains", []))
    r = response.lower()
    must = case.get("expected_contains", [])
    must_not = case.get("expected_not_contains", [])
    hits = [kw for kw in must if kw.lower() in r]
    misses = [kw for kw in must_not if kw.lower() in r]
    passed = (len(hits) == len(must)) and (len(misses) == 0)
    return passed, hits, misses


# ─── Главный цикл ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True, help="Имя архитектуры (для имени файла)")
    parser.add_argument("--dataset", default=os.path.join(os.path.dirname(__file__), "dataset.jsonl"))
    parser.add_argument("--out_dir", default=os.path.dirname(__file__))
    parser.add_argument("--limit", type=int, default=None, help="Прогнать только N кейсов")
    args = parser.parse_args()

    out_path = os.path.join(args.out_dir, f"eval_results_{args.arch}.jsonl")
    catalog = load_catalog()

    with open(args.dataset, encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        cases = cases[:args.limit]

    print(f"\n{'='*70}")
    print(f"Архитектура: {args.arch}")
    print(f"Датасет:     {args.dataset}  ({len(cases)} кейсов)")
    print(f"Результат -> {out_path}")
    print(f"{'='*70}\n")

    print("Инициализация бота...")
    bot = init_bot()
    print("Бот готов. Запуск прогона.\n")

    results = []

    for i, case in enumerate(cases):
        cid = case["id"]
        ctype = case.get("type", "single")
        gt = case.get("ground_truth_programs", [])
        eval_ret = case.get("evaluate_retrieval", False)

        print(f"[{i+1:02d}/{len(cases)}] {cid} ({ctype})...", end=" ", flush=True)

        t0 = time.time()
        exc = None
        llm_calls = None
        try:
            if ctype == "single":
                response, retrieved, llm_calls = run_single(bot, case["user_input"], catalog)
            elif ctype == "multi":
                response, retrieved, llm_calls = run_multi(bot, case["turns"], catalog)
            else:
                raise ValueError(f"Неизвестный тип: {ctype}")
        except Exception as e:
            response = f"[EXCEPTION] {type(e).__name__}: {e}"
            retrieved = []
            exc = traceback.format_exc()

        elapsed = round(time.time() - t0, 3)

        passed, hits, misses = auto_check(case, response)

        retrieval_metrics = {}
        if eval_ret and gt:
            retrieval_metrics = {
                "P@3": round(precision_at_k(retrieved, gt, 3), 4),
                "P@5": round(precision_at_k(retrieved, gt, 5), 4),
                "R@3": round(recall_at_k(retrieved, gt, 3), 4),
                "R@5": round(recall_at_k(retrieved, gt, 5), 4),
                "MRR": round(reciprocal_rank(retrieved, gt), 4),
                "nDCG@5": round(ndcg_at_k(retrieved, gt, 5), 4),
            }

        status = "v" if passed else "x"
        ret_str = ""
        if retrieval_metrics:
            ret_str = (f"  P@5={retrieval_metrics['P@5']:.2f}"
                       f" R@5={retrieval_metrics['R@5']:.2f}"
                       f" MRR={retrieval_metrics['MRR']:.2f}")
        llm_str = f"  LLM={llm_calls}" if llm_calls is not None else ""
        print(f"{status} {elapsed:.2f}s{ret_str}{llm_str}")
        if exc:
            print(f"        EXCEPTION: {response}")

        results.append({
            **case,
            "actual_response": response,
            "retrieved_programs": retrieved,
            "auto_passed": passed,
            "auto_hits": hits,
            "auto_misses": misses,
            "retrieval_metrics": retrieval_metrics,
            "elapsed_sec": elapsed,
            "llm_calls": llm_calls,
            "exception": exc,
            "arch": args.arch,
            "ts": time.time(),
        })

    # ─── Сводка ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"СВОДКА — {args.arch}")
    print(f"{'='*70}")

    total = len(results)
    pass_total = sum(1 for r in results if r["auto_passed"])
    probs = [r for r in results if r["problem_case"]]
    pass_probs = sum(1 for r in probs if r["auto_passed"])
    exceptions = [r["id"] for r in results if r["exception"]]
    times = [r["elapsed_sec"] for r in results]

    print(f"\n  Прохождение auto-check:")
    print(f"    Всего:        {pass_total}/{total}   ({100*pass_total/total:.1f}%)")
    print(f"    Проблемные:   {pass_probs}/{len(probs)}    ({100*pass_probs/max(len(probs),1):.1f}%)")
    if exceptions:
        print(f"    Исключения ({len(exceptions)}): {', '.join(exceptions[:8])}")

    ret_results = [r for r in results if r["retrieval_metrics"]]
    if ret_results:
        print(f"\n  Retrieval-метрики (среднее по {len(ret_results)} кейсам с GT):")
        for k in ["P@3", "P@5", "R@3", "R@5", "MRR", "nDCG@5"]:
            vals = [r["retrieval_metrics"][k] for r in ret_results]
            print(f"    {k:<8} = {mean(vals):.4f}")

    print(f"\n  Латентность (сек):")
    print(f"    median = {median(times):.2f}")
    sorted_t = sorted(times)
    p95 = sorted_t[int(len(sorted_t) * 0.95)] if len(sorted_t) > 1 else sorted_t[0]
    print(f"    p95    = {p95:.2f}")
    print(f"    mean   = {mean(times):.2f}")

    llm_counts = [r["llm_calls"] for r in results if r["llm_calls"] is not None]
    if llm_counts:
        print(f"\n  LLM-вызовы:")
        print(f"    total  = {sum(llm_counts)}")
        print(f"    mean   = {mean(llm_counts):.2f} на запрос")

    print(f"\n  По сценариям:")
    by_scenario = {}
    for r in results:
        s = r["scenario"]
        by_scenario.setdefault(s, {"total": 0, "passed": 0, "rm": []})
        by_scenario[s]["total"] += 1
        if r["auto_passed"]:
            by_scenario[s]["passed"] += 1
        if r["retrieval_metrics"]:
            by_scenario[s]["rm"].append(r["retrieval_metrics"])

    for s in sorted(by_scenario):
        d = by_scenario[s]
        pct = 100 * d["passed"] / d["total"]
        line = f"    {s}: {d['passed']:>2}/{d['total']:<2} ({pct:>5.1f}%)"
        if d["rm"]:
            r5 = mean(m["R@5"] for m in d["rm"])
            mrr = mean(m["MRR"] for m in d["rm"])
            line += f"   R@5={r5:.2f}   MRR={mrr:.2f}"
        print(line)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  Результаты записаны -> {out_path}\n")


if __name__ == "__main__":
    main()
