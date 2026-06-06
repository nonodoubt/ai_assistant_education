"""
services/rag/rag_chatbot.py — RAG чат-бот ДДТ «Союз» v3.

Логика обработки запроса (по приоритетам):

  A. МГНОВЕННЫЕ ответы (0 LLM)
     - приветствие → готовый текст
     - hardcoded ответы (площадки)

  B. ФИО педагога / название программы
     - детект по справочнику из БД
     - search_programs(БЕЗ возрастного фильтра) + 1 LLM для формулировки

  C. FAQ-паттерны
     - detect_faq_keywords → search_faq + 1 LLM

  D. ЛОКАЛЬНЫЙ СБОР данных (0 LLM)
     - quick_extract_age, quick_extract_location, detect_directions
     - обновляем self.collected_age, location, keywords

  E. ДОСТАТОЧНО данных для поиска?
     - есть направление + возраст → search + 1 LLM

  F. КОРОТКОЕ сообщение (< 6 слов), данных не хватает
     - уточняющий вопрос по шаблону (0 LLM)

  G. СЛОЖНОЕ сообщение (6+ слов), данных не хватает
     - LLM-экстрактор → попытка поиска → 1 LLM для ответа

Followup-запросы ("расскажи подробнее", "про X") — отдельная обработка:
- если в state уже есть направление → ищем без фильтра возраста
"""
import json, os, sys, re, sqlite3

from config.prompts import MODEL_PRIORITY, GREETING_RESPONSE, EXTRACTION_PROMPT, ANSWER_PROMPT
from services.api_key_manager.api_key_manager import (
    SmartKeyManager, Logger, call_with_cascade, TruncatedResponseError,
)
from services.preprocessor.preprocessor import (
    detect_greeting, detect_faq_keywords, quick_extract_age,
    quick_extract_location, detect_directions, is_gender_only,
    extract_significant_words, query_has_content,
)
from services.rag.search import (
    search_programs, search_faq, format_results_for_llm,
    format_results_markdown, SYNONYM_MAP, CATEGORY_MAP,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Определение дней недели в тексте
# ═══════════════════════════════════════════════════════════════════════════════

def _day_in_text(text_lower, forms):
    """
    True, если в тексте упомянут день. Сокращения (пн, вт, вс, чт...) матчатся
    только как ОТДЕЛЬНОЕ слово (двусторонняя граница), иначе «вс» ловит «все»,
    «чт» — «что», «ср» — «среда». Полные названия/стемы — по префиксу (ловят
    «вторникам», «четвергам», «воскресеньям» и др. словоформы).
    """
    for f in forms:
        pat = r'\b' + re.escape(f) + (r'\b' if len(f) <= 3 else '')
        if re.search(pat, text_lower):
            return True
    return False


# Формы дней недели для РАСПОЗНАВАНИЯ в запросе (единый источник). Стемы покрывают
# словоформы: «воскресен» → воскресенье/воскресеньям; «пятниц» → пятница/пятницу/
# пятницам. Для «среда» используем явные формы, т.к. стем «сред» ловит «среди».
_DAY_DETECT = [
    ('понедельник', ['понедельник', 'пн']),
    ('вторник', ['вторник', 'вт']),
    ('среда', ['среда', 'среду', 'среды', 'средам', 'ср']),
    ('четверг', ['четверг', 'чтеверг', 'чт']),
    ('пятница', ['пятниц', 'пт']),
    ('суббота', ['суббот', 'сб']),
    ('воскресенье', ['воскресен', 'вс']),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Загрузка справочников из БД (фамилии педагогов, названия программ)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_teacher_names(db_path):
    """
    Индекс педагогов: список кортежей (token, role, full_name).
    token — нормализованное слово ФИО (ё→е, lower); role — 'surname' для первого
    слова персоны, 'other' для имени/отчества. full_name — исходная строка из БД.
    Индексируем И фамилии, И имена, И отчества — чтобы находить педагога по
    «Сергей Петрович», а не только по фамилии.
    """
    # роли/направления, встречающиеся в поле teacher у комплексных программ
    _roles = {
        'вокал', 'хореография', 'хореографии', 'шахматы', 'изо', 'дпи', 'керамика',
        'английский', 'фото', 'видео', 'театральная', 'театр', 'студия', 'подготовка',
        'школе', 'школа', 'педагог', 'дизайн', 'экскурсоведение', 'каллиграфия',
        'моделирование', 'одежды', 'графика', 'творчества', 'мастерская',
    }
    tokens = []
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute('SELECT DISTINCT teacher FROM programs WHERE teacher IS NOT NULL')
        for row in cur.fetchall():
            full = row[0]
            # делим на персон по разделителям и роли
            for chunk in re.split(r'[,;]|\s-\s|\sи\s', full):
                words = [w.lower().replace('ё', 'е')
                         for w in re.findall(r'[А-Яа-яЁёA-Za-z\-]+', chunk)]
                words = [w for w in words if len(w) >= 3 and w not in _roles and w.isalpha()]
                if not words:
                    continue
                for i, w in enumerate(words):
                    role = 'surname' if i == 0 else 'other'
                    tokens.append((w, role, full))
        conn.close()
    except Exception:
        pass
    return tokens


def _common_prefix_len(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _load_program_names(db_path):
    """Загружает названия программ (lowercase, ё→е)."""
    names = {}
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute('SELECT DISTINCT name FROM programs')
        for row in cur.fetchall():
            original = row[0]
            m = re.search(r'[«"](.*?)[»"]', original)
            if m:
                clean = m.group(1).lower().replace('ё', 'е')
                names[clean] = original
            clean_full = original.lower().replace('ё', 'е')
            names[clean_full] = original
        conn.close()
    except Exception:
        pass
    return names


# ═══════════════════════════════════════════════════════════════════════════════
# RAG CHATBOT
# ═══════════════════════════════════════════════════════════════════════════════

class RAGChatbot:
    # Параметры
    COMPLEX_MESSAGE_WORDS = 6  # 6+ слов считается сложным
    # Порог сложного сообщения
    COMPLEX_MESSAGE_WORDS = 6

    def __init__(self, db_path, key_manager, logger, models=None):
        self.db_path = db_path
        self.key_manager = key_manager
        self.logger = logger
        self.models = models or MODEL_PRIORITY
        self.current_model = self.models[0]
        self.last_key_idx = None

        self.conversation_history = []
        self.collected_keywords = []
        self.collected_age = None
        self.collected_location = None
        self.collected_is_free = None
        self.collected_days = []  # запрошенные дни недели
        self.llm_calls = 0

        self._teacher_tokens = _load_teacher_names(db_path)
        self._program_names = _load_program_names(db_path)
        self._last_query = ""
        self.logger.debug("Бот инициализирован: %d токенов ФИО, %d названий программ" %
                          (len(self._teacher_tokens), len(self._program_names)))

    # ─── Утилиты ───

    def _call_llm(self, messages, system_prompt):
        self.llm_calls += 1
        result, model_used, key_idx = call_with_cascade(
            messages, system_prompt, self.key_manager, self.logger, self.models)
        if model_used != self.current_model:
            self.logger.debug("Модель: %s -> %s" % (self.current_model, model_used))
            self.current_model = model_used
        self.last_key_idx = key_idx
        return result

    def _add_to_history(self, role, text):
        self.conversation_history.append({"role": role, "parts": [{"text": text}]})
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

    def _finalize(self, response, category):
        """Логирует категорию и сохраняет в историю."""
        self.logger.debug("[категория: %s]" % category)
        self._add_to_history("model", response)
        return response

    def _is_followup(self, text, new_kws_added=0):
        """
        Followup = текущее сообщение НЕ добавило новых направлений.
        Значит пользователь уточняет предыдущий запрос.
        Работает для ЛЮБЫХ формулировок без словарей.
        """
        if not self.collected_keywords:
            return False
        return new_kws_added == 0

    def _has_direction(self):
        """Есть ли в collected_keywords направление (не возраст и не цена)?"""
        return any(
            kw.lower() not in ('бюджет', 'платно') and not re.match(r'^\d+', kw)
            for kw in self.collected_keywords)

    def _detect_teacher_or_program(self, text):
        """Ищет педагога (по фамилии/имени/отчеству) или название программы."""
        text_lower = text.lower().replace('ё', 'е')
        words = re.findall(r'[а-яеa-z\-]+', text_lower)
        _skip_q = {'записаться', 'записать', 'запиши', 'занятия', 'занятие',
                   'хочу', 'нужно', 'можно', 'педагог', 'педагогу', 'преподаватель'}

        # Голосование за полное ФИО: каждое слово запроса притягивается к токену
        # ФИО с самым длинным общим префиксом (различает «петровичу»→отчество
        # «Петрович» и фамилию «Петрова»). Тай-брейк — в пользу фамилии.
        votes = {}      # full_name -> накопленный score
        for w in words:
            if len(w) < 4 or w in _skip_q:
                continue
            best = None  # (prefix_len, role_rank, full)
            for tok, role, full in self._teacher_tokens:
                l = _common_prefix_len(w, tok)
                if l < 5:
                    continue
                if l < min(len(w), len(tok)) - 2:
                    continue
                if abs(len(w) - len(tok)) > 3:
                    continue
                role_rank = 1 if role == 'surname' else 0
                cand = (l, role_rank, full)
                if best is None or cand[:2] > best[:2]:
                    best = cand
            if best:
                votes[best[2]] = votes.get(best[2], 0) + best[0]
        if votes:
            full = max(votes, key=votes.get)
            return full, 'teacher'

        for prog_name in self._program_names:
            # Пропускаем «названия», совпадающие с обычным направлением
            # (керамика, английский, хореография…) — иначе путь поиска по названию
            # перехватывает запрос до сбора возраста. Названием считаем только
            # отличительные слова («Забава», «Рондо», «Белый ферзь»).
            if prog_name in SYNONYM_MAP or prog_name in CATEGORY_MAP:
                continue
            if prog_name in text_lower:
                return prog_name, 'program'

        return None, None

    # ─── Категория A: Мгновенные ответы ───

    def _try_instant(self, text):
        """Приветствие, hardcoded ответы."""
        if detect_greeting(text):
            return GREETING_RESPONSE

        faq_kws = detect_faq_keywords(text)
        if 'площадки' in faq_kws:
            return ("У ДДТ «Союз» две площадки:\n\n"
                    "**1. пр. Раевского, 5/2** — основная площадка\n\n"
                    "**2. пр. Тореза, 35/1** — дополнительная площадка\n\n"
                    "Если хотите узнать, какие программы доступны на конкретной площадке, "
                    "просто уточните!")
        return None

    # ─── Категория B: ФИО / название программы ───

    def _try_name_search(self, text):
        name, name_type = self._detect_teacher_or_program(text)
        if not name:
            return None

        self.logger.debug("Найдено %s: %s" % (name_type, name))
        if name not in [k.lower() for k in self.collected_keywords]:
            self.collected_keywords.append(name)

        programs = search_programs(
            self.db_path, [name], age=None, location=None, is_free=None,
            query_text=text)

        # Если ФИО педагога — оставляем ТОЛЬКО его программы
        if name_type == 'teacher' and programs:
            name_norm = name.lower().replace('ё', 'е')
            filtered = [p for p in programs
                        if p.get('teacher') and name_norm in p['teacher'].lower().replace('ё', 'е')]
            if filtered:
                programs = filtered

        # Если название программы — оставляем ТОЛЬКО программы где есть это слово в названии
        if name_type == 'program' and programs:
            name_norm = name.lower().replace('ё', 'е')
            filtered = [p for p in programs
                        if name_norm in p.get('name', '').lower().replace('ё', 'е')]
            if filtered:
                programs = filtered

        if not programs:
            return None

        return self._respond_with_programs(programs)

    # ─── Категория C: FAQ из БД ───

    # Порог уверенности для семантического FAQ-роутинга.
    # vec score = 1 - cosine_distance (нормированные эмбеддинги bge-m3 / e5).
    # 0.78 = очень похоже по смыслу. Эмпирически: "Где вы находитесь?" ↔ "где вы
    # находитесь" ~0.95; "Сколько стоит занятие?" ↔ "цена за месяц" ~0.82;
    # "Кто ты?" ↔ "ты бот?" ~0.80 (если в FAQ есть identity-запись).
    # Несвязанные запросы типа "мальчик 4 года шахматы" ↔ любой FAQ редко >0.7.
    FAQ_SEMANTIC_THRESHOLD = 0.78

    def _try_faq_search(self, text):
        """
        Гибридный FAQ-роутинг: keyword-якоря + семантический score.

        Решение, выдавать ли FAQ-ответ, принимается СЕМАНТИЧЕСКИ, а не по списку
        подстрок. Так покрываются формулировки, которые keyword-паттерны не
        ловят («где ВЫ находитесь», «можно с ребёнком на занятии», «вы кто»),
        и при этом не вываливаются программы на мета-вопросы.

        Условие выдачи FAQ:
          • есть keyword-матч в faq_kws → как раньше (быстрый путь);
          • ИЛИ топ-1 FAQ по вектору score ≥ FAQ_SEMANTIC_THRESHOLD
            И запрос не выглядит как поиск программы (нет распознанного
            направления, нет числового возраста).
        """
        faq_kws = detect_faq_keywords(text)

        # Площадки обработаны в категории A — не дублируем
        if 'площадки' in faq_kws:
            return None

        # Быстрый путь: keyword-якорь
        if faq_kws:
            faq_kws_sorted = sorted(faq_kws, key=len, reverse=True)
            all_faq = []
            seen_ids = set()
            for kw in faq_kws_sorted:
                items = search_faq(self.db_path, [kw], limit=2, query_text=text)
                for item in items:
                    if item['id'] not in seen_ids:
                        all_faq.append(item)
                        seen_ids.add(item['id'])
            if all_faq:
                self.logger.debug("FAQ keyword: %d по ключам %s"
                                  % (len(all_faq), faq_kws))
                return self._respond_with_programs([], all_faq[:3])

        # Семантический путь: запускаем ТОЛЬКО когда нет признаков поиска
        # программы. Иначе «4 года шахматы» может семантически зацепиться
        # за FAQ «С какого возраста можно записаться» — нежелательно.
        from services.preprocessor.preprocessor import (
            quick_extract_age, extract_significant_words,
        )
        has_direction = bool(detect_directions(text, SYNONYM_MAP, CATEGORY_MAP))
        has_age = quick_extract_age(text) is not None
        if has_direction or has_age:
            return None

        # Считаем семантическую близость к FAQ
        try:
            from services.rag.search import _vector_search_faq
            vec_scores = _vector_search_faq(self.db_path, text, limit=3)
        except Exception as e:
            self.logger.debug("FAQ vector недоступен: %s" % str(e)[:80])
            return None
        if not vec_scores:
            return None

        top_id, top_score = max(vec_scores.items(), key=lambda kv: kv[1])
        self.logger.debug("FAQ semantic: top score=%.3f (threshold=%.2f)"
                          % (top_score, self.FAQ_SEMANTIC_THRESHOLD))
        if top_score < self.FAQ_SEMANTIC_THRESHOLD:
            return None

        # Достаём FAQ-запись
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        ids = sorted(vec_scores.keys(), key=lambda i: -vec_scores[i])[:2]
        ph = ','.join('?' * len(ids))
        cur.execute('SELECT * FROM faq WHERE id IN (%s)' % ph, ids)
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        # Берём только записи выше порога
        faq_items = [r for r in rows
                     if vec_scores.get(r['id'], 0) >= self.FAQ_SEMANTIC_THRESHOLD]
        if not faq_items:
            return None
        # Сортируем в порядке убывания score
        faq_items.sort(key=lambda r: -vec_scores.get(r['id'], 0))
        self.logger.debug("FAQ semantic: выдано %d записей" % len(faq_items))
        return self._respond_with_programs([], faq_items[:2])

    # ─── Категория D: Локальный сбор ───

    def _collect_local_info(self, text):
        """Извлекает возраст, площадку, направления, дни. Возвращает кол-во НОВЫХ keywords."""
        from services.preprocessor.preprocessor import extract_all_ages, extract_age_proxy
        all_ages = extract_all_ages(text)
        if len(all_ages) == 1:
            self.collected_age = all_ages[0]
        elif len(all_ages) > 1:
            self.collected_age = None
            for a in all_ages:
                kw = "%d лет" % a
                if kw not in self.collected_keywords:
                    self.collected_keywords.append(kw)
            self.logger.debug("Несколько возрастов: %s" % all_ages)
        elif self.collected_age is None:
            # Явного числа нет — пробуем возрастное слово-прокси (подросток→14 и т.п.)
            proxy_age = extract_age_proxy(text)
            if proxy_age is not None:
                self.collected_age = proxy_age
                self.logger.debug("Возраст из слова-прокси: %d" % proxy_age)

        # Дни недели → в state
        DAYS = dict(_DAY_DETECT)
        text_lower = text.lower()
        for day, forms in DAYS.items():
            if _day_in_text(text_lower, forms):
                if day not in self.collected_days:
                    self.collected_days.append(day)
                    self.logger.debug("Сохранён день: %s" % day)

        loc = quick_extract_location(text)
        if loc:
            self.collected_location = loc

        local_kws = detect_directions(text, SYNONYM_MAP, CATEGORY_MAP)
        new_count = 0
        for kw in local_kws:
            if kw not in [k.lower() for k in self.collected_keywords]:
                self.collected_keywords.append(kw)
                new_count += 1

        if local_kws or all_ages or loc:
            self.logger.debug("Локально: age=%s, loc=%s, kws=%s, days=%s (новых: %d)" %
                              (self.collected_age, self.collected_location,
                               self.collected_keywords, self.collected_days, new_count))
        return new_count

    # ─── Категория E: Достаточно данных? ───

    def _has_enough_info(self, text, new_kws_added=0):
        has_direction = self._has_direction()
        has_age = self.collected_age is not None
        if self._is_followup(text, new_kws_added) and has_direction:
            return True
        return has_direction and has_age

    def _search_and_respond(self, text, new_kws_added=0):
        """Поиск + ответ."""
        programs, faq_items = self._do_search(text)

        if not programs:
            # Интерес распознан, но под возраст/площадку/дни ничего нет — честный
            # ответ ВАЖНЕЕ, чем случайный FAQ-фолбэк.
            notice = self._age_mismatch_notice(text)
            if notice:
                return notice
            if not faq_items:
                return None

        return self._respond_with_programs(programs, faq_items)

    @staticmethod
    def _age_fits(program, age):
        amin, amax = program.get('age_min'), program.get('age_max')
        if amin is not None and amin > age:
            return False
        if amax is not None and amax < age:
            return False
        return True

    _DAY_STEMS = [
        ('понедельник', ['понедельник', 'пн']),
        ('вторник', ['вторник', 'вт']),
        ('среда', ['сред', 'ср']),
        ('четверг', ['четверг', 'чтеверг', 'чт']),
        ('пятница', ['пятниц', 'пт']),
        ('суббота', ['суббот', 'сб']),
        ('воскресенье', ['воскресен', 'вс']),
    ]

    def _interest_label(self):
        """Человекочитаемый тег интереса для сообщений о несоответствии."""
        for k in self.collected_keywords:
            if k.lower() not in ('бюджет', 'платно') and not re.match(r'^\d+\s*лет', k):
                return k
        kws = getattr(self, '_last_search_kws', None) or []
        for k in kws:
            if k.lower() not in ('бюджет', 'платно') and not re.match(r'^\d+\s*лет', k):
                return k
        return "этим направлением"

    def _prog_days(self, program):
        """Канонические дни недели из расписания программы."""
        sched = (program.get('schedule') or '').lower()
        out = []
        for day, stems in self._DAY_STEMS:
            if any(s in sched for s in stems) and day not in out:
                out.append(day)
        return out

    def _age_mismatch_notice(self, query_text):
        """
        Честный ответ, когда программы по интересу ЕСТЬ, но ни одна не подходит
        под заданный фильтр. Проверяем по очереди: возраст → площадка → дни.
        Иначе None (обычный ход).
        """
        search_kws = getattr(self, '_last_search_kws', None)
        if not search_kws:
            return None
        # Все программы по интересу без фильтров
        base = search_programs(self.db_path, search_kws, age=None,
                               location=None, is_free=None, query_text=query_text)
        if not base:
            return None  # такого направления нет вообще — обычный фолбэк

        tag = self._interest_label()

        # 1) Возраст
        age = self.collected_age
        if age is not None:
            fits_age = [p for p in base if self._age_fits(p, age)]
            if not fits_age:
                ranges = []
                for p in base:
                    a = (p.get('age_str') or '').strip()
                    if a and a not in ranges:
                        ranges.append(a)
                ranges_str = ', '.join(ranges) if ranges else 'другой возраст'
                self.logger.debug("Несоответствие ВОЗРАСТА (%s): есть на %s" % (age, ranges_str))
                return ("К сожалению, у нас нет такой программы для этого возраста. "
                        "Есть для возраста: %s." % ranges_str)
        else:
            fits_age = base

        # 2) Площадка
        if self.collected_location and fits_age:
            loc_norm = self.collected_location.lower().replace('ё', 'е')
            loc_ok = [p for p in fits_age
                      if (p.get('location_norm') or '').lower().replace('ё', 'е') == loc_norm]
            if not loc_ok:
                locs = []
                for p in fits_age:
                    l = (p.get('location') or p.get('location_norm') or '').strip()
                    if l and l not in locs:
                        locs.append(l)
                locs_str = ', '.join(locs) if locs else 'других площадках'
                self.logger.debug("Несоответствие ПЛОЩАДКИ: есть на %s" % locs_str)
                return ("К сожалению, программы, связанные с «%s», есть на площадке: %s."
                        % (tag, locs_str))

        # 3) Дни недели
        if self.collected_days and fits_age:
            day_ok = [p for p in fits_age if self._runs_on_days(p, self.collected_days)]
            if not day_ok:
                avail_days = []
                for p in fits_age:
                    for d in self._prog_days(p):
                        if d not in avail_days:
                            avail_days.append(d)
                days_str = ', '.join(avail_days) if avail_days else 'другим дням'
                self.logger.debug("Несоответствие ДНЕЙ: есть по %s" % days_str)
                return ("К сожалению, программы, связанные с «%s», есть по другим дням: %s."
                        % (tag, days_str))

        return None

    def _do_search(self, query_text):
        """Выполняет поиск, возвращает (programs, faq_items)."""
        # search_kws = словарные ключи (бустер) + значимые слова запроса (backbone).
        # Словарь синонимов покрывает не все формулировки, поэтому содержательные
        # слова пользователя ВСЕГДА доходят до контентного поиска (FTS + вектора).
        search_kws = list(self.collected_keywords)

        sig_words = extract_significant_words(query_text or '')
        existing_lower = {k.lower() for k in search_kws}
        for w in sig_words:
            if w not in existing_lower:
                search_kws.append(w)
                existing_lower.add(w)
        if sig_words:
            self.logger.debug("Backbone-слова из запроса: %s" % sig_words)

        # Дни недели → в state
        DAYS = dict(_DAY_DETECT)
        text_lower = (query_text or '').lower()
        for k in self.collected_keywords:
            text_lower += ' ' + k.lower()
        for day, forms in DAYS.items():
            if _day_in_text(text_lower, forms):
                if day not in self.collected_days:
                    self.collected_days.append(day)
                    self.logger.debug("Сохранён день: %s" % day)

        all_day_words = set()
        for forms in DAYS.values():
            for f in forms:
                all_day_words.add(f)

        # Удаляем дни из search_kws
        search_kws = [k for k in search_kws
                      if k.lower() not in all_day_words]
        self._last_search_kws = list(search_kws)

        exclude_pre = getattr(self, '_exclude_preschool', False)

        has_real_kws = any(k.lower() not in ('бюджет', 'платно')
                          and not re.match(r'^\d+\s*лет', k)
                          for k in search_kws)

        if not has_real_kws and self.collected_age:
            programs = search_programs(self.db_path, [],
                                       age=self.collected_age,
                                       location=self.collected_location,
                                       is_free=self.collected_is_free,
                                       query_text=query_text,
                                       exclude_preschool=exclude_pre,
                                       limit=20)
            self.logger.debug("Поиск по возрасту %d: %d программ" %
                              (self.collected_age, len(programs)))
            programs = self._filter_by_days(programs, self.collected_days)
            return programs, []

        # Основной поиск — СТРОГО по возрасту. Если ребёнок не попадает в возрастной
        # диапазон программы, он не может записаться — поэтому такие программы не
        # предлагаем (честный ответ про несоответствие возраста формирует
        # _age_mismatch_notice выше по стеку).
        programs = search_programs(self.db_path, search_kws,
                                   age=self.collected_age,
                                   location=self.collected_location,
                                   is_free=self.collected_is_free,
                                   query_text=query_text,
                                   exclude_preschool=exclude_pre)
        if not programs and self.collected_is_free is not None:
            # Релаксируем ТОЛЬКО цену (площадку оставляем строгой — про неё бот
            # честно сообщит через _age_mismatch_notice). Возраст соблюдаем.
            programs = search_programs(self.db_path, search_kws,
                                       age=self.collected_age,
                                       location=self.collected_location,
                                       query_text=query_text,
                                       exclude_preschool=exclude_pre)

        # Защитный фолбэк: возраст известен, КОНКРЕТНОЕ направление не распознано,
        # а основной поиск пуст (значит backbone-слова — это шум/возрастные формы,
        # не давшие контентных совпадений). Показываем программы по возрасту, а не
        # сообщаем «ничего нет». Не срабатывает, если направление распознано —
        # тогда отрабатывает честный ответ про несоответствие возраста.
        has_direction = any(
            k.lower() not in ('бюджет', 'платно') and not re.match(r'^\d+\s*лет', k)
            for k in self.collected_keywords)
        if not programs and self.collected_age and not has_direction:
            programs = search_programs(self.db_path, [],
                                       age=self.collected_age,
                                       location=self.collected_location,
                                       is_free=self.collected_is_free,
                                       exclude_preschool=exclude_pre, limit=20)
            self.logger.debug("Фолбэк-листинг по возрасту %d: %d программ" %
                              (self.collected_age, len(programs)))

        programs = self._filter_by_days(programs, self.collected_days)

        faq_items = []
        if not programs:
            faq_items = search_faq(self.db_path, search_kws, query_text=query_text)

        self.logger.debug("Поиск: %d программ, %d FAQ (kws: %s, дни: %s)" %
                          (len(programs), len(faq_items), search_kws, self.collected_days))
        return programs, faq_items

    def _runs_on_days(self, program, requested_days):
        """Идёт ли программа хотя бы в один из запрошенных дней (устойчиво к опечаткам)."""
        prog_days = self._prog_days(program)
        return any(d in prog_days for d in requested_days)

    def _filter_by_days(self, programs, requested_days):
        """
        Оставляет программы, идущие в запрошенные дни. Если совпадений нет —
        возвращает ПУСТО (не «все подряд»), чтобы бот честно сказал про другие дни
        через _age_mismatch_notice.
        """
        if not requested_days or not programs:
            return programs
        filtered = [p for p in programs if self._runs_on_days(p, requested_days)]
        self.logger.debug("Фильтр по дням %s: %d -> %d программ" %
                          (requested_days, len(programs), len(filtered)))
        return filtered

    # ─── Категория F: Уточнение по шаблону (0 LLM) ───

    def _ask_clarification(self, text):
        has_age = self.collected_age is not None
        has_direction = self._has_direction()

        if is_gender_only(text) and not has_age:
            return "Подскажите, сколько лет ребёнку?"

        text_lower = text.lower().strip()

        # "Все" / "какие есть" / "что есть" — явный запрос показать всё
        SHOW_ALL_PATTERNS = [
            'все', 'всё', 'все занятия', 'всё занятия', 'все направления',
            'список', 'покажи все', 'покажи всё', 'покажи список',
            'какие есть', 'что есть', 'какие занятия', 'какие программы',
            'что доступно', 'какие направления', 'что предлагаете',
            'перечисли', 'все варианты',
        ]
        if any(p in text_lower for p in SHOW_ALL_PATTERNS):
            return None  # сигнал показать все

        # Если возраст есть — проверяем сколько программ для него
        if has_age and not has_direction:
            programs = search_programs(self.db_path, [],
                                       age=self.collected_age, limit=10)
            if len(programs) <= 3:
                self.logger.debug("Для %d лет всего %d программ — показываем все" %
                                  (self.collected_age, len(programs)))
                return None
            return ("Для %d лет у нас много интересного! А какие у ребёнка интересы? "
                    "Может быть: танцы, рисование, вокал, шахматы, театр, керамика, "
                    "компьютерный дизайн?" % self.collected_age)

        if not has_age and has_direction:
            return "А сколько лет вашему ребёнку?"

        return ("Подскажите, пожалуйста, сколько лет вашему ребёнку и какие у него интересы? "
                "Например: танцы, вокал, рисование, шахматы, театр, керамика.")

    # ─── Категория G: LLM-экстрактор для сложных сообщений ───

    def _handle_complex(self, user_message):
        """Сложное сообщение → LLM-экстрактор → search + LLM."""
        ext = self._extract_keywords_via_llm(user_message)

        if ext is None:
            return self._fallback_response()

        self._update_collected_info(ext)
        intent = ext.get("intent", "unclear")
        self.logger.debug("LLM-экстрактор: intent=%s, kws=%s, age=%s" %
                          (intent, self.collected_keywords, self.collected_age))

        if intent == "greeting":
            return GREETING_RESPONSE

        # ПРИОРИТЕТ: "какие есть", "все" — показываем всё для возраста
        clarification = self._ask_clarification(user_message)
        if clarification is None and self.collected_age:
            # SHOW_ALL pattern matched ИЛИ мало программ для возраста
            # Принудительно ищем БЕЗ keywords (только возраст)
            saved_kws = list(self.collected_keywords)
            self.collected_keywords = [k for k in self.collected_keywords
                                       if re.match(r'^\d+\s*лет', k)
                                       or k.lower() in ('бюджет', 'платно')]
            response = self._search_and_respond(user_message)
            self.collected_keywords = saved_kws
            if response:
                return response

        # Достаточно данных для обычного поиска
        if self._has_enough_info(user_message) or self._has_direction():
            response = self._search_and_respond(user_message)
            if response:
                return response

        return clarification or ext.get("clarifying_question") or self._fallback_response()

    def _extract_keywords_via_llm(self, user_message):
        hist = "".join(
            "%s: %s\n" % ("Пользователь" if m['role'] == 'user' else "Ассистент",
                          m['parts'][0]['text'])
            for m in self.conversation_history[-10:])
        try:
            raw = self._call_llm(
                [{"role": "user", "parts": [{"text":
                    "ИСТОРИЯ:\n%s\n\nСООБЩЕНИЕ: %s" % (hist, user_message)}]}],
                EXTRACTION_PROMPT)
        except Exception as e:
            self.logger.debug("LLM-экстрактор упал: %s" % str(e)[:100])
            return None

        try:
            cleaned = re.sub(r'```json\s*|```\s*', '', raw)
            return json.loads(cleaned.strip())
        except json.JSONDecodeError:
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    pass
        return None

    def _update_collected_info(self, ext):
        if ext is None:
            return
        for kw in ext.get("key_words", []):
            if kw.lower() not in [k.lower() for k in self.collected_keywords]:
                self.collected_keywords.append(kw)
        if ext.get("age") is not None:
            self.collected_age = ext["age"]
        if ext.get("location") is not None:
            self.collected_location = ext["location"]
        if ext.get("is_free") is not None:
            self.collected_is_free = ext["is_free"]

    def _respond_with_programs(self, programs, faq_items=None):
        """
        Формулирует ответ по найденным программам через LLM.
        Каскад моделей сам переключается между моделями/ключами при ошибках,
        поэтому шаблонный ответ используется ТОЛЬКО если ВСЕ модели реально
        недоступны (крайний случай).

        Побочный эффект: записывает в self.last_served_directions список
        направлений тех программ, что попали в ответ. Используется server.py
        для логирования (фильтр админки «по направлениям» работает по этому
        полю, а не по keywords запроса).
        """
        if faq_items is None:
            faq_items = []
        faq_items = faq_items[:2]
        # Запоминаем направления реально выданных программ (нормализация: strip)
        served = []
        for p in (programs or []):
            d = (p.get('direction') or '').strip() if isinstance(p, dict) else ''
            if d:
                served.append(d)
        # Накапливаем за весь process_message (вдруг будет несколько вызовов)
        prev = getattr(self, 'last_served_directions', None) or []
        self.last_served_directions = sorted(set(prev + served))
        try:
            context = format_results_for_llm(programs, faq_items)
            return self._call_llm(self.conversation_history,
                                  ANSWER_PROMPT.format(context=context))
        except Exception as e:
            self.logger.debug("Все модели недоступны, крайний фолбэк: %s" % str(e)[:80])
            return format_results_markdown(programs, faq_items)

    # ─── Fallback ───

    def _fallback_response(self):
        if self.collected_age or self.collected_keywords:
            programs, faq_items = self._do_search("")
            if programs or faq_items:
                return self._respond_with_programs(programs, faq_items)
        return ("Подскажите, пожалуйста, сколько лет вашему ребёнку и какие у него интересы? "
                "Например: танцы, вокал, рисование, шахматы, театр.")

    # ─── Главный метод ───

    def _reset_all(self):
        """Полный сброс собранного состояния и истории (новый ребёнок)."""
        self.collected_keywords = []
        self.collected_age = None
        self.collected_location = None
        self.collected_is_free = None
        self.collected_days = []
        self.conversation_history = []

    def _maybe_reset_state(self, text):
        """
        Управление памятью между ходами:
        • Новый ОТЛИЧНЫЙ возраст после ответа → полный сброс (это другой ребёнок).
          «девочка 13, лепить» → ответ → «а для мальчика 14?» → ищем заново под 14.
        • Новое НАПРАВЛЕНИЕ без смены возраста → забываем прежнее направление,
          возраст сохраняем. «…лепить» → ответ → «а танцы?» → «танцы» для того же возраста.
        """
        if not self.conversation_history:
            return  # первый запрос — сбрасывать нечего

        from services.preprocessor.preprocessor import extract_all_ages
        new_ages = extract_all_ages(text)
        prev_age = self.collected_age

        # 1) Новый отличный возраст → полный сброс
        if len(new_ages) == 1 and prev_age is not None and new_ages[0] != prev_age:
            self.logger.debug("Новый возраст %d (был %s) → полный сброс памяти"
                              % (new_ages[0], prev_age))
            self._reset_all()
            return

        # 2) Новое направление без смены возраста → заменить направление
        new_dirs = detect_directions(text, SYNONYM_MAP, CATEGORY_MAP)
        if new_dirs:
            old_dirs = [k for k in self.collected_keywords
                        if k.lower() not in ('бюджет', 'платно')
                        and not re.match(r'^\d+\s*лет', k)]
            new_norm = [d.lower() for d in new_dirs]
            old_norm = [d.lower() for d in old_dirs]
            if old_dirs and any(d not in old_norm for d in new_norm):
                self.logger.debug("Новое направление %s → забываю прежнее %s"
                                  % (new_dirs, old_dirs))
                self.collected_keywords = [
                    k for k in self.collected_keywords
                    if k.lower() in ('бюджет', 'платно')
                    or re.match(r'^\d+\s*лет', k)]

    def process_message(self, user_message):
        text = user_message.strip()
        # Сбрасываем трекер выданных направлений на каждый ход (server.py читает
        # его после возврата из process_message и передаёт в log_message).
        self.last_served_directions = []
        # Сброс/замена состояния при новом возрасте или новом направлении (до истории)
        self._maybe_reset_state(text)
        self._add_to_history("user", user_message)
        self._last_query = user_message

        try:
            # ─── A: Мгновенные ответы (0 LLM) ───
            response = self._try_instant(text)
            if response:
                return self._finalize(response, "A:instant")

            # ─── B: ФИО педагога / название программы (1 LLM) ───
            response = self._try_name_search(text)
            if response:
                return self._finalize(response, "B:name_search")

            # ─── C: FAQ из БД (1 LLM) ───
            response = self._try_faq_search(text)
            if response:
                return self._finalize(response, "C:faq")

            # ─── D: Локальный сбор данных (0 LLM) ───
            new_kws = self._collect_local_info(text)

            # ─── E: Достаточно данных? (1 LLM) ───
            if self._has_enough_info(text, new_kws):
                response = self._search_and_respond(text, new_kws)
                if response:
                    return self._finalize(response, "E:search")

            # Направление распознано, но возраста нет → уточняем возраст, а НЕ
            # вываливаем весь список направления. Программы зависят от возраста
            # (керамика 4-11, английский 7-8 и т.д.), поэтому без возраста ответ
            # был бы недостоверным. Не срабатывает, если возраст(а) уже известны
            # или пользователь явно просит «показать всё».
            if self._has_direction() and self.collected_age is None:
                has_multi_age = any(re.match(r'^\d+\s*лет', k)
                                    for k in self.collected_keywords)
                wants_all = any(p in text.lower() for p in
                                ('все', 'всё', 'любые', 'любое', 'список', 'покажи'))
                if not has_multi_age and not wants_all:
                    return self._finalize(
                        "Подскажите, пожалуйста, сколько лет вашему ребёнку?",
                        "F:ask_age")

            # ─── E2: Вектора могут найти даже без keyword-матча ───
            is_bare_gender = is_gender_only(text)
            is_bare_number = bool(re.match(r'^\d{1,2}$', text.strip()))
            age_extracted = quick_extract_age(text) is not None

            # Значимые слова — той же функцией, что и backbone в _do_search
            # (фильтрует стоп-, служебные, возрастные слова: «для», «при», «про»…).
            from services.preprocessor.preprocessor import extract_significant_words
            significant_words = extract_significant_words(text)

            # "Только возраст без значимых слов" → F уточнит интересы
            has_only_age_no_dir = (age_extracted and not self._has_direction()
                                   and new_kws == 0 and len(significant_words) == 0)

            is_single_word_query = (len(text.split()) <= 2 and len(text.strip()) >= 4
                                    and not is_bare_gender and not is_bare_number
                                    and not has_only_age_no_dir)

            # Followup: направление было РАНЕЕ
            is_followup_with_direction = (self._has_direction() and new_kws == 0
                                          and not is_bare_gender and not is_bare_number)

            # Есть значимые слова + возраст, но синонимы не сматчились → пробуем vector
            has_unknown_significant = (len(significant_words) > 0 and new_kws == 0
                                       and not is_bare_gender and not is_bare_number)

            should_try_vector = (is_followup_with_direction
                                 or is_single_word_query
                                 or has_unknown_significant)

            if should_try_vector:
                programs, faq_items = self._do_search(text)
                if programs or faq_items:
                    return self._finalize(
                        self._respond_with_programs(programs, faq_items),
                        "E2:vector_found")

            # Возраст указан, направления и значимых слов нет → ВСЕГДА уточняем
            # интересы, независимо от длины фразы. «найди другому ребёнку 7 лет» —
            # это 6 слов, но по сути только возраст: показывать весь список или
            # гадать вектором не нужно, надо спросить интерес (как для «4 года»).
            if has_only_age_no_dir:
                clar = self._ask_clarification(text)
                if clar is None:  # программ для возраста мало → показать их
                    programs, faq_items = self._do_search(text)
                    if programs or faq_items:
                        return self._finalize(
                            self._respond_with_programs(programs, faq_items),
                            "F:show_all")
                return self._finalize(
                    clar or ("А какие у ребёнка интересы? Например: танцы, вокал, "
                             "рисование, шахматы, театр, керамика."),
                    "F:clarify_interest")

            # ─── F: Короткое сообщение, данных не хватает (0 LLM) ───
            word_count = len(text.lower().split())
            if word_count < self.COMPLEX_MESSAGE_WORDS:
                clarification = self._ask_clarification(text)
                if clarification is None:
                    programs, faq_items = self._do_search(text)
                    if programs or faq_items:
                        return self._finalize(
                            self._respond_with_programs(programs, faq_items),
                            "F:show_all")
                return self._finalize(clarification, "F:clarify_short")

            # ─── G: Сложное сообщение (1-2 LLM) ───
            response = self._handle_complex(user_message)
            return self._finalize(response, "G:complex")

        except Exception as e:
            self.logger.debug("Необработанная ошибка: %s" % str(e))
            return self._finalize(self._fallback_response(), "fallback")

    def reset(self):
        self.conversation_history.clear()
        self.collected_keywords.clear()
        self.collected_age = self.collected_location = self.collected_is_free = None
        self.collected_days = []
        self._exclude_preschool = False
        self._last_query = ""