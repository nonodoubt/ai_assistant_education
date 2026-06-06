"""
services/api_key_manager/api_key_manager.py

SmartKeyManager v3:
- Отдельный учёт использования по комбинации (ключ, модель)
- Cooldown по комбинации (ключ, модель), а не по ключу целиком
- Это значит: если ключ #1 исчерпал лимит на 2.5-flash-lite, его ещё можно
  использовать с 2.0-flash-lite

Структура key_stats.json:
{
  "0": {  // индекс ключа
    "key_hint": "AIza...",
    "by_model": {
      "gemini-2.5-flash-lite": {"uses": 10, "successes": 8, "failures": 2, "disabled_until": null},
      "gemini-2.0-flash-lite": {"uses": 3, "successes": 3, "failures": 0, "disabled_until": null}
    },
    "total_uses": 13,
    "total_successes": 11,
    "total_failures": 2
  }
}
"""

import json
import time
import os
import requests
from datetime import datetime


class APIError(Exception):
    pass

class RateLimitError(APIError):
    pass

class ModelOverloadedError(APIError):
    pass

class ModelNotFoundError(APIError):
    pass

class TruncatedResponseError(Exception):
    """Ответ обрезан из-за MAX_TOKENS."""
    pass

class TransportError(APIError):
    """Сетевой/таймаут на стороне клиента — обычно проблема МОДЕЛИ, не ключа.
    Эквивалент 503: другой ключ не поможет, надо переключать модель."""
    pass


class Logger:
    def __init__(self, mode="user"):
        self.mode = mode

    def debug(self, msg):
        if self.mode == "dev":
            print("  [DEBUG] %s" % msg)

    def info(self, msg):
        print(msg)

    def warn(self, msg):
        if self.mode == "dev":
            print("  [WARN] %s" % msg)


class SmartKeyManager:
    """
    v3: отдельный учёт по (ключ, модель).
    """

    def __init__(self, keys, logger, cooldown_seconds=60, stats_file=None):
        self.keys = keys
        self.logger = logger
        self.cooldown = cooldown_seconds
        self.total_keys = len(keys)
        self._next_index = 0
        self.stats_file = stats_file or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "key_stats.json")

        # stats[key_idx]["by_model"][model_name] = {uses, successes, failures, disabled_until}
        self.stats = {}
        for i in range(len(keys)):
            self.stats[i] = {
                "key_hint": keys[i][:8] + "...",
                "by_model": {},
                "total_uses": 0,
                "total_successes": 0,
                "total_failures": 0,
            }
        self._load_stats()

    def _load_stats(self):
        if not os.path.exists(self.stats_file):
            return
        try:
            with open(self.stats_file, 'r') as f:
                saved = json.load(f)
            for k, v in saved.items():
                idx = int(k)
                if idx in self.stats:
                    # Backward compat: старый формат имел uses/successes/failures прямо в корне
                    if "by_model" in v:
                        self.stats[idx]["by_model"] = v["by_model"]
                        self.stats[idx]["total_uses"] = v.get("total_uses", 0)
                        self.stats[idx]["total_successes"] = v.get("total_successes", 0)
                        self.stats[idx]["total_failures"] = v.get("total_failures", 0)
        except Exception as e:
            self.logger.warn("Ошибка загрузки stats: %s" % e)

    def _save_stats(self):
        try:
            data = {}
            for i, s in self.stats.items():
                # Не сохраняем disabled_until (он временный)
                by_model_clean = {}
                for m, info in s["by_model"].items():
                    by_model_clean[m] = {
                        "uses": info.get("uses", 0),
                        "successes": info.get("successes", 0),
                        "failures": info.get("failures", 0),
                    }
                data[str(i)] = {
                    "key_hint": s["key_hint"],
                    "by_model": by_model_clean,
                    "total_uses": s["total_uses"],
                    "total_successes": s["total_successes"],
                    "total_failures": s["total_failures"],
                    "last_updated": datetime.now().isoformat(),
                }
            with open(self.stats_file, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.warn("Ошибка сохранения stats: %s" % e)

    def _get_model_stats(self, key_idx, model):
        """Возвращает или создаёт запись для (ключ, модель)."""
        if model not in self.stats[key_idx]["by_model"]:
            self.stats[key_idx]["by_model"][model] = {
                "uses": 0, "successes": 0, "failures": 0, "disabled_until": None,
            }
        return self.stats[key_idx]["by_model"][model]

    def _is_available_for_model(self, key_idx, model):
        """Доступен ли ключ для конкретной модели?"""
        ms = self._get_model_stats(key_idx, model)
        until = ms.get("disabled_until")
        if until is None:
            return True
        if time.time() >= until:
            ms["disabled_until"] = None
            return True
        return False

    def get_key_for_model(self, model):
        """
        Возвращает (api_key, key_idx) — ключ доступный для модели.
        Round-robin по доступным для этой модели ключам.
        """
        checked = 0
        while checked < self.total_keys:
            idx = self._next_index % self.total_keys
            self._next_index = (self._next_index + 1) % self.total_keys
            if self._is_available_for_model(idx, model):
                ms = self._get_model_stats(idx, model)
                ms["uses"] += 1
                self.stats[idx]["total_uses"] += 1
                return self.keys[idx], idx
            checked += 1

        # Все ключи на cooldown для этой модели — сбрасываем
        self.logger.warn("Все ключи на cooldown для %s. Сбрасываю..." % model)
        for i in range(self.total_keys):
            ms = self._get_model_stats(i, model)
            ms["disabled_until"] = None

        idx = self._next_index % self.total_keys
        self._next_index = (self._next_index + 1) % self.total_keys
        ms = self._get_model_stats(idx, model)
        ms["uses"] += 1
        self.stats[idx]["total_uses"] += 1
        return self.keys[idx], idx

    # Backward compat
    def get_key(self):
        """Старый метод без модели — оставлен для обратной совместимости."""
        return self.get_key_for_model("default")

    def report_success(self, key_idx, model="default"):
        ms = self._get_model_stats(key_idx, model)
        ms["successes"] += 1
        self.stats[key_idx]["total_successes"] += 1
        self._save_stats()

    def report_failure(self, key_idx, error="", model="default", is_rate_limit=False):
        ms = self._get_model_stats(key_idx, model)
        ms["failures"] += 1
        self.stats[key_idx]["total_failures"] += 1
        # Cooldown ставим только для rate-limit (429), не для всех ошибок
        if is_rate_limit:
            ms["disabled_until"] = time.time() + self.cooldown
            self.logger.debug("Ключ #%d на cooldown для %s (%ds)" %
                              (key_idx + 1, model, self.cooldown))
        self._save_stats()

    def get_stats_report(self):
        """Подробный отчёт: по каждому ключу — статистика по моделям."""
        lines = ["=" * 70]
        lines.append("Статистика API ключей (по моделям)")
        lines.append("=" * 70)

        for i in range(self.total_keys):
            s = self.stats[i]
            lines.append("\nКлюч #%d [%s]" % (i + 1, s["key_hint"]))
            lines.append("  Всего: исп=%d, усп=%d, ош=%d" %
                          (s["total_uses"], s["total_successes"], s["total_failures"]))

            if s["by_model"]:
                for model, info in sorted(s["by_model"].items()):
                    status = "OK"
                    until = info.get("disabled_until")
                    if until and time.time() < until:
                        remaining = int(until - time.time())
                        status = "cooldown %ds" % remaining
                    lines.append("    %s: исп=%d усп=%d ош=%d [%s]" %
                                  (model, info["uses"], info["successes"],
                                   info["failures"], status))
            else:
                lines.append("    (не использовался)")

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# GEMINI API
# ═══════════════════════════════════════════════════════════════════════════════

def _build_payload(messages, system_prompt):
    return {
        "contents": messages,
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 12000},
    }


def _check_error(status_code, resp_text):
    try:
        error_msg = json.loads(resp_text).get("error", {}).get("message", resp_text[:300])
    except Exception:
        error_msg = resp_text[:300]
    if status_code == 404:
        raise ModelNotFoundError("404: %s" % error_msg)
    elif status_code == 429:
        raise RateLimitError("429: %s" % error_msg)
    elif status_code == 503:
        raise ModelOverloadedError("503: %s" % error_msg)
    else:
        raise APIError("%d: %s" % (status_code, error_msg))


def call_gemini(messages, system_prompt, api_key, model, request_timeout=12):
    url = ("https://generativelanguage.googleapis.com/v1beta/models/%s"
           ":generateContent?key=%s") % (model, api_key)
    try:
        resp = requests.post(url, json=_build_payload(messages, system_prompt),
                             timeout=request_timeout)
    except requests.exceptions.Timeout as e:
        # Read/Connect timeout — это проблема МОДЕЛИ (или сети), не ключа.
        # Бросаем TransportError → caller переключит модель, а не будет долбить
        # ту же модель другими ключами (тратя на каждом по 12с).
        raise TransportError("timeout: %s" % str(e)[:120])
    except requests.exceptions.ConnectionError as e:
        raise TransportError("connection: %s" % str(e)[:120])
    if resp.status_code == 200:
        data = resp.json()
        candidate = data["candidates"][0]
        text = candidate["content"]["parts"][0]["text"]
        finish = candidate.get("finishReason", "")
        if finish == "MAX_TOKENS":
            print("[api] ВНИМАНИЕ: ответ обрезан (MAX_TOKENS)")
            raise TruncatedResponseError("MAX_TOKENS")
        return text
    _check_error(resp.status_code, resp.text)


def call_with_cascade(messages, system_prompt, key_manager, logger, models,
                      max_time=25):
    """
    Каскадный вызов:
    1. Для модели — выбираем СЛУЧАЙНЫЙ доступный ключ (не по порядку)
    2. При 429 — другой случайный ключ для ТОЙ ЖЕ модели
    3. При 503 / таймауте / ошибке сети → СРАЗУ следующая модель
       (другой ключ той же модели не поможет — это проблема модели/сети)
    4. После 3 неудач подряд → следующая модель
    5. Общий timeout max_time секунд (по умолчанию 25с — должно хватить
       на 4 модели по ~6с с per-request timeout=12с)
    """
    import time as _time
    import random as _random
    t_start = _time.time()
    last_error = None
    MAX_TRIES_PER_MODEL = 3

    for model in models:
        tries = 0
        tried_keys = set()

        while tries < MAX_TRIES_PER_MODEL:
            if _time.time() - t_start > max_time:
                raise Exception("Timeout %.0fs. Последняя ошибка: %s" %
                                (_time.time() - t_start, last_error))

            # Доступные ключи для этой модели (не на cooldown, ещё не пробовали)
            available = [i for i in range(key_manager.total_keys)
                         if i not in tried_keys
                         and key_manager._is_available_for_model(i, model)]

            if not available:
                logger.debug("Нет доступных ключей для %s -> следующая модель" % model)
                break

            idx = _random.choice(available)
            tried_keys.add(idx)
            tries += 1

            api_key = key_manager.keys[idx]
            ms = key_manager._get_model_stats(idx, model)
            ms["uses"] += 1
            key_manager.stats[idx]["total_uses"] += 1

            try:
                result = call_gemini(messages, system_prompt, api_key, model)
                key_manager.report_success(idx, model=model)
                logger.debug("OK: модель=%s, ключ=#%d (попытка %d)" %
                              (model, idx + 1, tries))
                return result, model, idx
            except TruncatedResponseError:
                raise  # пропускаем наверх — ответ получен, но обрезан
            except ModelNotFoundError as e:
                last_error = e
                logger.debug("Модель %s не найдена -> следующая модель" % model)
                break
            except RateLimitError as e:
                last_error = e
                key_manager.report_failure(idx, str(e), model=model, is_rate_limit=True)
                logger.debug("429 ключ #%d для %s (попытка %d/%d)" %
                              (idx + 1, model, tries, MAX_TRIES_PER_MODEL))
                continue
            except ModelOverloadedError as e:
                last_error = e
                logger.debug("Модель %s перегружена (503) -> следующая модель" % model)
                break
            except TransportError as e:
                # Таймаут / разрыв соединения = проблема модели или сети.
                # Другой ключ той же модели почти наверняка тоже зависнет
                # (мы это и видели в логах: 30с×N ключей). Переходим к
                # следующей модели сразу.
                last_error = e
                key_manager.report_failure(idx, str(e), model=model, is_rate_limit=False)
                logger.debug("Сеть/таймаут на %s ключ #%d -> следующая модель"
                              % (model, idx + 1))
                break
            except (APIError, Exception) as e:
                last_error = e
                key_manager.report_failure(idx, str(e), model=model, is_rate_limit=False)
                logger.debug("Ошибка ключ #%d: %s" % (idx + 1, str(e)[:100]))
                continue

    raise Exception("Все модели и ключи исчерпаны. Последняя ошибка: %s" % last_error)


# ═══════════════════════════════════════════════════════════════════════════════
# ЗАГРУЗКА КЛЮЧЕЙ
# ═══════════════════════════════════════════════════════════════════════════════

def load_api_keys(project_root):
    accounts_path = os.path.join(
        project_root, "services", "api_key_manager", "api_accounts.xlsx")
    keys = []
    if os.path.exists(accounts_path):
        try:
            import pandas as pd
            df = pd.read_excel(accounts_path)
            key_col = None
            for col in df.columns:
                if 'key' in col.lower() or 'api' in col.lower():
                    key_col = col
                    break
            if key_col is None:
                key_col = df.columns[-1]
            for val in df[key_col].dropna():
                k = str(val).strip()
                if k and len(k) > 10:
                    keys.append(k)
        except Exception as e:
            print("Ошибка чтения ключей: %s" % e)
    env_key = os.environ.get("GEMINI_API_KEY", "")
    if env_key and env_key not in keys:
        keys.append(env_key)
    return keys


def view_stats(stats_file=None):
    """CLI-просмотр статистики."""
    if stats_file is None:
        stats_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "key_stats.json")
    if not os.path.exists(stats_file):
        print("Файл статистики не найден: %s" % stats_file)
        return
    with open(stats_file, 'r') as f:
        data = json.load(f)
    print("=" * 70)
    print("  Статистика API ключей по моделям")
    print("=" * 70)
    for key_id, info in sorted(data.items(), key=lambda x: int(x[0])):
        idx = int(key_id) + 1
        hint = info.get("key_hint", "???")
        total_uses = info.get("total_uses", 0)
        total_succ = info.get("total_successes", 0)
        total_fail = info.get("total_failures", 0)
        print("\nКлюч #%d [%s]: всего исп=%d усп=%d ош=%d" %
              (idx, hint, total_uses, total_succ, total_fail))

        by_model = info.get("by_model", {})
        if by_model:
            for model, m_info in sorted(by_model.items()):
                uses = m_info.get("uses", 0)
                succ = m_info.get("successes", 0)
                fail = m_info.get("failures", 0)
                rate = "%d%%" % (succ * 100 // uses) if uses > 0 else "-"
                print("    %s: исп=%d усп=%d (%s) ош=%d" %
                      (model, uses, succ, rate, fail))
    print("\n" + "=" * 70)


if __name__ == "__main__":
    import sys
    view_stats(sys.argv[1] if len(sys.argv) > 1 else None)