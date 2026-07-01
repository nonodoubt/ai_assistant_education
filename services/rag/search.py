"""
services/rag/search.py — Гибридный поиск: FTS5/теги + sqlite-vec + reranker.

Модели (bge-m3, reranker) передаются ИЗВНЕ через init_models().
Это позволяет предзагрузить их при старте приложения.
"""

import sqlite3
import json
import re
import os
import sys
import struct
import time
from typing import Optional

try:
    from services.preprocessor.preprocessor import query_has_content
except Exception:  # pragma: no cover — на случай иного контекста запуска
    def query_has_content(text):
        return bool(text and text.strip())

try:
    from config.prompts import EMBEDDING_MODEL
except Exception:  # pragma: no cover
    EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

# ═══════════════════════════════════════════════════════════════════════════════
# НАСТРОЙКИ ГИБРИДНОГО РАНЖИРОВАНИЯ (вынесены для калибровки на eval-датасете)
# ═══════════════════════════════════════════════════════════════════════════════

# Веса объединения keyword/vector в итоговом score
W_KEYWORD = 0.4          # вклад keyword-сигнала (нормируется делением на 3)
W_VECTOR = 0.6           # вклад векторной близости (cosine, 0..1)

# Гейт релевантности
# Когда ЕСТЬ keyword-матчи: vector-only кандидат проходит, если его score
# не ниже этой доли от топового (т.е. он почти так же близок семантически).
VEC_ONLY_KEEP_RATIO = 0.7
# Когда keyword-матчей НЕТ вообще (чистый семантический поиск по незнакомому
# словарю слову): держим только то, что близко к топу, и жёстко ограничиваем
# число результатов — чтобы не вываливать весь корпус.
PURE_VEC_KEEP_RATIO = 0.85
PURE_VEC_MAX_RESULTS = 6

# ═══════════════════════════════════════════════════════════════════════════════
# СЛОВАРИ
# ═══════════════════════════════════════════════════════════════════════════════

def _load_synonyms(filepath=None):
    if filepath is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for path in [os.path.join(script_dir, '..', 'db', 'synonyms.json'),
                     os.path.join(script_dir, 'synonyms.json')]:
            if os.path.exists(path):
                filepath = path
                break
    if not filepath or not os.path.exists(filepath):
        raise FileNotFoundError("synonyms.json не найден")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return ({k: v for k, v in data.get('synonym_map', {}).items() if k != '_comment'},
            {k: v for k, v in data.get('category_map', {}).items() if k != '_comment'})

SYNONYM_MAP, CATEGORY_MAP = _load_synonyms()

_RE_NONWORD = re.compile(r'[^\w\s]')


def _norm(text):
    return text.lower().replace('ё', 'е')


def normalize_keywords(keywords):
    normalized = []
    for kw in keywords:
        kw_lower = _norm(kw.strip())
        if kw_lower in CATEGORY_MAP:
            normalized.extend(CATEGORY_MAP[kw_lower])
        elif kw_lower in SYNONYM_MAP:
            result = SYNONYM_MAP[kw_lower]
            normalized.append(result)
            # Если результат составной ("бальные танцы") — добавляем и отдельные слова
            if ' ' in result:
                normalized.extend(result.split())
        else:
            normalized.append(kw_lower)
            # Также разбиваем составные входные слова
            if ' ' in kw_lower:
                normalized.extend(kw_lower.split())
    return list(set(normalized))


def extract_age_from_keywords(keywords):
    for kw in keywords:
        for n in re.findall(r'\d+', kw):
            age = int(n)
            if 2 <= age <= 18:
                return age
    return None


def extract_location_from_keywords(keywords):
    for kw in keywords:
        if 'раевск' in kw.lower():
            return 'Раевского'
        if 'торез' in kw.lower():
            return 'Тореза'
    return None


def _clean_kw(kw):
    return _RE_NONWORD.sub('', kw)


def _build_fts_terms(normalized):
    terms = []
    for kw in normalized:
        clean = _clean_kw(kw)
        if clean and len(clean) >= 2:
            terms.append('"%s"' % clean)
    return terms


# ═══════════════════════════════════════════════════════════════════════════════
# МОДЕЛИ — глобальные, устанавливаются через init_models()
# ═══════════════════════════════════════════════════════════════════════════════

_embedding_model = None
_reranker_model = None
_vec_available = None
_models_initialized = False


def init_models(embedding_model=None, reranker_model=None):
    """
    Устанавливает предзагруженные модели.
    Вызывается из app.py после @st.cache_resource.
    """
    global _embedding_model, _reranker_model, _models_initialized
    _embedding_model = embedding_model
    _reranker_model = reranker_model
    _models_initialized = True
    print("[search] Модели инициализированы: emb=%s, reranker=%s" %
          (type(embedding_model).__name__ if embedding_model else 'None',
           type(reranker_model).__name__ if reranker_model else 'None'))


def _get_embedding_model():
    global _embedding_model, _models_initialized
    if _models_initialized:
        return _embedding_model
    # Fallback: ленивая загрузка если init_models не вызван
    if _embedding_model is None:
        try:
            print("[search] Ленивая загрузка %s..." % EMBEDDING_MODEL)
            t = time.time()
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            print("[search] %s загружен за %.1fs" % (EMBEDDING_MODEL, time.time() - t))
        except Exception as e:
            print("[search] Эмбеддинг-модель недоступна: %s" % e)
            _embedding_model = False
    return _embedding_model if _embedding_model is not False else None


def _get_reranker():
    global _reranker_model, _models_initialized
    if _models_initialized:
        return _reranker_model
    if _reranker_model is None:
        try:
            print("[search] Ленивая загрузка reranker...")
            t = time.time()
            from sentence_transformers import CrossEncoder
            _reranker_model = CrossEncoder("BAAI/bge-reranker-v2-m3")
            print("[search] reranker загружен за %.1fs" % (time.time() - t))
        except Exception as e:
            print("[search] reranker недоступен: %s" % e)
            _reranker_model = False
    return _reranker_model if _reranker_model is not False else None


def _is_vec_available(db_path):
    global _vec_available
    if _vec_available is not None:
        return _vec_available
    try:
        import sqlite_vec
        conn = sqlite3.connect(db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM vec_programs")
        count = cur.fetchone()[0]
        conn.close()
        _vec_available = count > 0
        print("[search] sqlite-vec: %s (%d векторов)" %
              ("OK" if _vec_available else "пусто", count))
    except Exception as e:
        print("[search] sqlite-vec недоступен: %s" % e)
        _vec_available = False
    return _vec_available


def _serialize_f32(vector):
    return struct.pack("%sf" % len(vector), *vector)


# ═══════════════════════════════════════════════════════════════════════════════
# KEYWORD SEARCH
# ═══════════════════════════════════════════════════════════════════════════════

def _keyword_search_programs(cur, keywords, normalized, limit=10):
    tag_kw = [kw for kw in normalized if kw not in ('бюджет', 'платно') and not re.match(r'^\d+$', kw)]
    tag_results = set()
    if tag_kw:
        ph = ','.join('?' * len(tag_kw))
        cur.execute('SELECT DISTINCT program_id FROM program_tags WHERE tag IN (%s)' % ph, tag_kw)
        tag_results = {r[0] for r in cur.fetchall()}

    fts_results = set()
    fts_terms = _build_fts_terms(normalized)
    if fts_terms:
        try:
            cur.execute('SELECT rowid FROM programs_fts WHERE programs_fts MATCH ? ORDER BY rank LIMIT ?',
                        (' OR '.join(fts_terms), limit * 2))
            fts_results = {r[0] for r in cur.fetchall()}
        except sqlite3.OperationalError:
            pass

    all_ids = tag_results | fts_results

    # LIKE-фоллбэк по реальному тексту программ (если FTS+теги ничего не дали)
    like_results = set()
    if not all_ids:
        all_terms = list(set(normalized + [_norm(k.strip()) for k in keywords]))
        for kw in all_terms:
            if len(kw) >= 3:
                cur.execute('SELECT id FROM programs WHERE search_text LIKE ? LIMIT ?',
                            ('%%%s%%' % kw, limit))
                like_results.update(r[0] for r in cur.fetchall())
        all_ids |= like_results

    # Скоринг: совпадение по РЕАЛЬНОМУ ТЕКСТУ программы (FTS/LIKE) — надёжный
    # сигнал (вес 2). Тег — лишь мягкий бустер (вес 1), т.к. данные тегов могут
    # быть зашумлены (напр. танцевальные программы помечены тегом «вокал»).
    # Поэтому одного тега недостаточно, чтобы гарантировать попадание в выдачу.
    scores = {}
    for pid in all_ids:
        s = 0
        if pid in fts_results or pid in like_results:
            s += 2
        if pid in tag_results:
            s += 1
        scores[pid] = s
    return scores


# ═══════════════════════════════════════════════════════════════════════════════
# VECTOR SEARCH
# ═══════════════════════════════════════════════════════════════════════════════

def _vector_search_programs(db_path, query_text, limit=10):
    model = _get_embedding_model()
    if model is None or not _is_vec_available(db_path):
        return {}
    try:
        import sqlite_vec
        t = time.time()
        query_emb = model.encode(["query: " + query_text], normalize_embeddings=True)[0]
        print("[search] Эмбеддинг запроса: %.2fs" % (time.time() - t))

        conn = sqlite3.connect(db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        cur = conn.cursor()

        cur.execute("""
            SELECT rowid, distance FROM vec_programs
            WHERE embedding MATCH ? ORDER BY distance LIMIT ?
        """, (_serialize_f32(query_emb), limit))

        results = {}
        for row in cur.fetchall():
            results[row[0]] = max(0, 1.0 - row[1])
        conn.close()
        # Это пул кандидатов (k ближайших), а НЕ итоговые совпадения. kNN всегда
        # возвращает свои top-k; что из них релевантно — решает гейт ниже.
        print("[search] Вектор: %d кандидатов в пуле (до фильтра релевантности)"
              % len(results))
        return results
    except Exception as e:
        print("[search] Ошибка векторного поиска: %s" % e)
        return {}


def _vector_search_faq(db_path, query_text, limit=5):
    model = _get_embedding_model()
    if model is None or not _is_vec_available(db_path):
        return {}
    try:
        import sqlite_vec
        query_emb = model.encode(["query: " + query_text], normalize_embeddings=True)[0]
        conn = sqlite3.connect(db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        cur = conn.cursor()
        cur.execute("""
            SELECT rowid, distance FROM vec_faq
            WHERE embedding MATCH ? ORDER BY distance LIMIT ?
        """, (_serialize_f32(query_emb), limit))
        results = {row[0]: max(0, 1.0 - row[1]) for row in cur.fetchall()}
        conn.close()
        return results
    except Exception:
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
# RERANKER
# ═══════════════════════════════════════════════════════════════════════════════

def _rerank(query, candidates, limit=5):
    """Реранкер отключён (слишком медленный на CPU). Простая сортировка по score."""
    return candidates[:limit]


# ═══════════════════════════════════════════════════════════════════════════════
# ГЛАВНЫЕ ФУНКЦИИ
# ═══════════════════════════════════════════════════════════════════════════════

def search_programs(db_path, keywords, age=None, location=None,
                    is_free=None, limit=15, query_text=None,
                    exclude_preschool=False, age_tolerance=0):
    t_start = time.time()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    normalized = normalize_keywords(keywords)
    if age is None:
        age = extract_age_from_keywords(keywords)
    if location is None:
        location = extract_location_from_keywords(keywords)
    if is_free is None:
        if 'бюджет' in normalized:
            is_free = True
        elif 'платно' in normalized:
            is_free = False

    # Возрастная ветка-листинг срабатывает ТОЛЬКО когда в запросе нет ни
    # словарных ключей, ни содержательных слов (например, голое «10 лет» или
    # вызов из _ask_clarification без query_text). Если пользователь выразил
    # интерес — пусть даже не распознанный словарём — идём в гибридный поиск.
    real_kws = [k for k in normalized if k not in ('бюджет', 'платно')]
    if not real_kws and not query_has_content(query_text) and age is not None:
        conditions = ['age_min <= ?', 'age_max >= ?']
        params = [age, age]
        if location:
            conditions.append('location_norm = ?')
            params.append(location)
        if is_free is not None:
            conditions.append('is_free = ?')
            params.append(1 if is_free else 0)
        if exclude_preschool:
            conditions.append('age_max >= 8')

        sql = 'SELECT * FROM programs WHERE ' + ' AND '.join(conditions) + ' ORDER BY name'
        cur.execute(sql, params)
        results = [dict(r) for r in cur.fetchall()[:limit]]
        conn.close()
        print("[search] По возрасту %d: %d программ (%.2fs)" %
              (age, len(results), time.time() - t_start))
        return results

    if query_text is None:
        query_text = ' '.join(keywords)

    # 1. Keyword search
    kw_scores = _keyword_search_programs(cur, keywords, normalized, limit=limit * 2)
    print("[search] Keyword: %d результатов" % len(kw_scores))

    # 2. Vector search (пул кандидатов; на маленьком корпусе незачем тащить почти
    #    весь набор — берём top-k, гейт ниже отфильтрует нерелевантное)
    vec_scores = _vector_search_programs(db_path, query_text, limit=limit)

    # 3. Объединение
    all_ids = set(kw_scores.keys()) | set(vec_scores.keys())
    if not all_ids:
        conn.close()
        print("[search] Ничего не найдено (%.2fs)" % (time.time() - t_start))
        return []

    combined_scores = {}
    for pid in all_ids:
        kw_s = kw_scores.get(pid, 0)
        vec_s = vec_scores.get(pid, 0)
        if vec_scores:
            combined = W_KEYWORD * (kw_s / 3.0) + W_VECTOR * vec_s
        else:
            combined = kw_s
        combined_scores[pid] = combined

    sorted_ids = sorted(combined_scores.keys(), key=lambda x: -combined_scores[x])
    top_ids = sorted_ids[:limit * 2]

    # 4. Забираем строки кандидатов БЕЗ ограничений (возраст/локация/цена
    #    применяются ПОСЛЕ гейта релевантности — см. ниже).
    ph = ','.join('?' * len(top_ids))
    cur.execute('SELECT * FROM programs WHERE id IN (%s)' % ph, top_ids)
    results = [dict(r) for r in cur.fetchall()]
    conn.close()

    for r in results:
        r['_score'] = combined_scores.get(r['id'], 0)
    results.sort(key=lambda x: -x['_score'])

    # 5. Фильтр релевантности (единый принцип для всех режимов):
    #  • Совпадение по реальному тексту программы (FTS/LIKE, kw_s>=2) — надёжный
    #    якорь точности, оставляем всегда.
    #  • Только тег (kw_s==1) или только вектор — слабый сигнал: оставляем лишь
    #    если score близок к топовому (т.е. подтверждён семантикой/контекстом).
    #  • Если надёжных якорей нет вовсе (чистая семантика по незнакомому слову) —
    #    более строгий порог и жёсткий лимит, чтобы не вываливать весь корпус.
    #
    # ВАЖНО: гейт идёт ДО фильтра по возрасту. Иначе возрастной фильтр может
    # выкинуть все контентные якоря (напр. керамика 7-11 при запросе на 12 лет),
    # top_score схлопнется на векторный шум, и сервис подсунет нерелевантные
    # программы, совпавшие лишь по возрасту/дню. Сначала — что релевантно, потом —
    # подходит ли это по возрасту.
    if results:
        top_score = results[0].get('_score', 0)
        strong = [r for r in results if kw_scores.get(r['id'], 0) >= 2]
        weak = [r for r in results if kw_scores.get(r['id'], 0) < 2]

        if strong:
            weak_kept = [r for r in weak
                         if r.get('_score', 0) >= top_score * VEC_ONLY_KEEP_RATIO]
            results = strong + weak_kept
        elif top_score > 0:
            weak_kept = [r for r in weak
                         if r.get('_score', 0) >= top_score * PURE_VEC_KEEP_RATIO]
            results = weak_kept[:PURE_VEC_MAX_RESULTS]

        results.sort(key=lambda x: -x.get('_score', 0))
        n_anchor = len([r for r in results if kw_scores.get(r['id'], 0) >= 2])
        print("[search] После гейта: %d релевантных (%d по тексту/FTS + %d по тегу/вектору)"
              % (len(results), n_anchor, len(results) - n_anchor))

    # 6. Ограничения (возраст/площадка/цена) применяются к УЖЕ релевантному
    #    набору. Если по интересу ничего не подходит под возраст — честно вернём
    #    пусто (оркестратор обработает: уточнит или сообщит, что нет вариантов),
    #    а не будем добивать нерелевантными программами.
    def _passes(r):
        if age is not None:
            amin, amax = r.get('age_min'), r.get('age_max')
            if amin is not None and amin > age + age_tolerance:
                return False
            if amax is not None and amax < age - age_tolerance:
                return False
        if exclude_preschool:
            amax = r.get('age_max')
            if amax is not None and amax < 8:
                return False
        if location and r.get('location_norm') != location:
            return False
        if is_free is not None and r.get('is_free') != (1 if is_free else 0):
            return False
        return True

    before = len(results)
    results = [r for r in results if _passes(r)]
    if before != len(results):
        print("[search] Ограничения (возраст/площадка/цена): %d -> %d" %
              (before, len(results)))

    # 5. Reranker
    if len(results) > 1 and query_text:
        results = _rerank(query_text, results, limit=limit)
    else:
        results = results[:limit]

    print("[search] Итого: %d программ (%.2fs)" % (len(results), time.time() - t_start))
    return results


def search_faq(db_path, keywords, limit=3, query_text=None):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    normalized = normalize_keywords(keywords)
    if query_text is None:
        query_text = ' '.join(keywords)

    fts_terms = _build_fts_terms(normalized)
    kw_ids = set()
    if fts_terms:
        try:
            cur.execute('SELECT rowid FROM faq_fts WHERE faq_fts MATCH ? ORDER BY rank LIMIT ?',
                        (' OR '.join(fts_terms), limit * 2))
            kw_ids = {r[0] for r in cur.fetchall()}
        except sqlite3.OperationalError:
            pass

    if not kw_ids:
        all_terms = set(normalized + [_norm(k.strip()) for k in keywords])
        expanded = set()
        for term in all_terms:
            expanded.add(term)
            for word in term.split():
                if len(word) >= 3:
                    expanded.add(word)
        for kw in expanded:
            if len(kw) >= 3:
                cur.execute('SELECT id FROM faq WHERE search_text LIKE ? LIMIT ?',
                            ('%%%s%%' % kw, limit))
                kw_ids.update(r[0] for r in cur.fetchall())
            if len(kw_ids) >= limit:
                break

    vec_scores = _vector_search_faq(db_path, query_text, limit=limit * 2)

    all_faq_ids = kw_ids | set(vec_scores.keys())
    if not all_faq_ids:
        conn.close()
        return []

    ph = ','.join('?' * len(all_faq_ids))
    cur.execute('SELECT * FROM faq WHERE id IN (%s)' % ph, list(all_faq_ids))
    results = [dict(r) for r in cur.fetchall()]

    for r in results:
        r['_score'] = (2 if r['id'] in kw_ids else 0) + vec_scores.get(r['id'], 0)
    results.sort(key=lambda x: -x['_score'])

    conn.close()
    return results[:limit]


def format_results_for_llm(programs, faq_items):
    """Контекст для LLM (сырой, с разделителями)."""
    parts = []
    if programs:
        parts.append("=== НАЙДЕННЫЕ ПРОГРАММЫ ===\n")
        for i, p in enumerate(programs, 1):
            parts.append("--- Программа %d ---" % i)
            parts.append(p.get('chunk_text', ''))
            parts.append("")
    if faq_items:
        parts.append("=== FAQ ===\n")
        for item in faq_items:
            parts.append("В: %s" % item['question'])
            parts.append("О: %s" % item['answer'])
            parts.append("")
    if not parts:
        return "Ничего не найдено в базе знаний."
    return '\n'.join(parts)


def format_results_markdown(programs, faq_items):
    """
    Красивый markdown БЕЗ LLM.
    Объединяет программы с одинаковым названием (разные площадки/расписание).
    """
    parts = []
    has_preschool = False

    if programs:
        parts.append("Вот что я нашёл:\n")

        # Группируем по названию
        from collections import OrderedDict
        grouped = OrderedDict()
        for p in programs:
            name = p.get('name', '?')
            if name not in grouped:
                grouped[name] = []
            grouped[name].append(p)

        for name, group in grouped.items():
            parts.append("**%s**" % name)
            first = group[0]

            if first.get('direction'):
                parts.append("- Направление: %s" % first['direction'])
            if first.get('results'):
                desc = first['results'].strip()
                sentences = desc.split('.')
                short = '. '.join(sentences[:2]).strip()
                if short and not short.endswith('.'):
                    short += '.'
                parts.append("- Чему научится: %s" % short)
            if first.get('age_str'):
                parts.append("- Возраст: %s" % first['age_str'])

            # Расписание и площадки — перечисляем все варианты
            if len(group) > 1:
                parts.append("- Варианты расписания:")
                for p in group:
                    loc = p.get('location', '')
                    sched = p.get('schedule', '')
                    if loc or sched:
                        parts.append("  - %s: %s" % (loc, sched))
            else:
                if first.get('schedule'):
                    parts.append("- Расписание: %s" % first['schedule'])
                if first.get('location'):
                    parts.append("- Площадка: %s" % first['location'])

            if first.get('price'):
                price = first['price']
                if 'бюджет' in price.lower() and 'беспл' not in price.lower():
                    price = price + ' (бесплатно)'
                parts.append("- Стоимость: %s" % price)
            if first.get('requirements'):
                parts.append("- *Что нужно: %s*" % first['requirements'])
            if first.get('teacher'):
                parts.append("- Педагог: %s" % first['teacher'])
            signup = (first.get('signup_url') or '').strip()
            if signup:
                if signup.startswith('http://') or signup.startswith('https://'):
                    parts.append("- [Записаться](%s)" % signup)
                else:
                    # в поле записи не ссылка, а текст (например «по телефону …»)
                    parts.append("- Запись: %s" % signup)

            for p in group:
                try:
                    if p.get('age_max') and int(p['age_max']) <= 7:
                        has_preschool = True
                except (ValueError, TypeError):
                    pass

            parts.append("")

    if faq_items:
        if programs:
            parts.append("---\n")
        for item in faq_items:
            parts.append("**%s**\n" % item.get('question', ''))
            parts.append("%s\n" % item.get('answer', ''))

    if has_preschool:
        parts.append("\n**Записаться на занятия:** https://forms.gle/NeUTe4nh5PQDvFMg7")
        parts.append("**Более подробная информация по телефону:** 8 995 834 09 94 (ПН-ПТ с 11 до 19ч)")

    if not parts:
        return ("К сожалению, ничего не найдено. Уточните на нашем сайте "
                "https://unionddt.ru/roditelyam-i-detyam/ "
                "или по телефону: 8 995 834 09 94 (ПН-ПТ с 11 до 19ч).")

    return '\n'.join(parts)