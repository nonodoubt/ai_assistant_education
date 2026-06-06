"""
build_db.py — Создаёт SQLite базу данных из programs.xlsx и FAQ.docx.
Включает FTS5 полнотекстовый индекс для поиска.
"""

import sqlite3
import pandas as pd
import json
import re
import os
import sys


def parse_age_range(age_str):
    """Извлекает min_age и max_age из строки вроде '7-17 лет' или '3-4 года'."""
    if not age_str or not isinstance(age_str, str):
        return None, None
    nums = re.findall(r'\d+', age_str)
    if len(nums) >= 2:
        return int(nums[0]), int(nums[1])
    elif len(nums) == 1:
        return int(nums[0]), int(nums[0])
    return None, None


def normalize_location(loc):
    """Нормализует название площадки."""
    if not loc or not isinstance(loc, str):
        return None
    loc = loc.strip().lower()
    if 'раевск' in loc:
        return 'Раевского'
    elif 'торез' in loc:
        return 'Тореза'
    return loc


def _normalize_text(text):
    """Нормализация для поиска: lowercase + ё→е."""
    return text.lower().replace('ё', 'е')


def build_search_text(row):
    """Собирает текст для полнотекстового поиска из всех полей строки."""
    parts = []
    for col in ['Направление', 'Название коллектива', 'Возраст',
                 'Платное/Бюджет', 'Расписание', 'Площадка',
                 'Что нужно для занятий', 'Чему ребенок научится/ что изучит?',
                 'Педагог', 'теги']:
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            parts.append(str(val).strip())
    text = ' '.join(parts)
    return _normalize_text(text)


def build_chunk_text(row):
    """Формирует человекочитаемый чанк для передачи в LLM."""
    parts = [f"Программа: {row['Название коллектива']}"]

    field_map = {
        'Направление': 'Направление',
        'Возраст': 'Возраст',
        'Площадка': 'Адрес',
        'Расписание': 'Расписание',
        'Платное/Бюджет': 'Стоимость',
        'Что нужно для занятий': 'Что нужно для занятий',
        'Чему ребенок научится/ что изучит?': 'Чему научится ребёнок',
        'Педагог': 'Педагог',
        'Информация о педагоге': 'О педагоге',
    }

    for field, label in field_map.items():
        val = row.get(field)
        if pd.notna(val) and str(val).strip():
            parts.append(f"{label}: {val}")

    url = row.get('Ссылка на запись (навигатор)')
    if pd.notna(url) and str(url).strip():
        parts.append(f"Ссылка для записи: {url}")

    return '\n'.join(parts)


def load_faq(filepath):
    """Читаем FAQ из docx, разбиваем на пары вопрос-ответ."""
    from docx import Document
    doc = Document(filepath)
    full_text = '\n'.join(p.text for p in doc.paragraphs if p.text.strip())

    # Разбиваем по вопросам (строки с ** жирным **)
    faq_items = []
    current_q = None
    current_a_lines = []

    for line in full_text.split('\n'):
        line = line.strip()
        if not line:
            continue
        # Определяем вопросы по наличию "?" или паттерну "**...**"
        clean = line.replace('*', '').strip()
        if '?' in line and len(clean) > 10:
            # Сохраняем предыдущий Q/A
            if current_q and current_a_lines:
                faq_items.append({
                    'question': current_q,
                    'answer': '\n'.join(current_a_lines)
                })
            current_q = clean
            current_a_lines = []
        elif current_q:
            current_a_lines.append(line.replace('*', '').strip())

    # Последний Q/A
    if current_q and current_a_lines:
        faq_items.append({
            'question': current_q,
            'answer': '\n'.join(current_a_lines)
        })

    return faq_items


def create_database(programs_path, faq_path, db_path='knowledge.db'):
    """Основная функция: создаёт БД, заполняет данные, строит FTS5 индекс."""
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # ─── Таблица программ ───
    cur.execute('''
        CREATE TABLE programs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            direction TEXT,
            age_str TEXT,
            age_min INTEGER,
            age_max INTEGER,
            price TEXT,
            is_free INTEGER,
            schedule TEXT,
            location TEXT,
            location_norm TEXT,
            requirements TEXT,
            results TEXT,
            teacher TEXT,
            teacher_info TEXT,
            signup_url TEXT,
            chunk_text TEXT,
            search_text TEXT
        )
    ''')

    # ─── Таблица FAQ ───
    cur.execute('''
        CREATE TABLE faq (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            search_text TEXT
        )
    ''')

    # ─── Таблица тегов/ключевых слов для программ ───
    cur.execute('''
        CREATE TABLE program_tags (
            program_id INTEGER,
            tag TEXT,
            FOREIGN KEY (program_id) REFERENCES programs(id)
        )
    ''')

    # ─── Загружаем программы ───
    df = pd.read_excel(programs_path)
    print(f"Загружено {len(df)} строк из Excel")

    # Словарь синонимов для тегов
    # Загружаем словарь тегов из JSON
    synonyms_path = os.path.join(os.path.dirname(programs_path), 'synonyms.json')
    if not os.path.exists(synonyms_path):
        # Пробуем рядом со скриптом
        synonyms_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'synonyms.json')
    
    with open(synonyms_path, 'r', encoding='utf-8') as f:
        syn_data = json.load(f)
    TAG_SYNONYMS = {k: v for k, v in syn_data.get('tag_synonyms', {}).items() if k != '_comment'}

    for idx, row in df.iterrows():
        age_min, age_max = parse_age_range(row.get('Возраст'))
        location_norm = normalize_location(row.get('Площадка'))
        price_str = str(row.get('Платное/Бюджет', '')).strip()
        is_free = 1 if 'бюджет' in price_str.lower() else 0
        chunk = build_chunk_text(row)
        search = build_search_text(row)

        cur.execute('''
            INSERT INTO programs
            (name, direction, age_str, age_min, age_max, price, is_free,
             schedule, location, location_norm, requirements, results,
             teacher, teacher_info, signup_url, chunk_text, search_text)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ''', (
            str(row.get('Название коллектива', '')).strip(),
            str(row.get('Направление', '')).strip(),
            str(row.get('Возраст', '')).strip(),
            age_min, age_max,
            price_str,
            is_free,
            str(row.get('Расписание', '')).strip() if pd.notna(row.get('Расписание')) else None,
            str(row.get('Площадка', '')).strip() if pd.notna(row.get('Площадка')) else None,
            location_norm,
            str(row.get('Что нужно для занятий', '')).strip() if pd.notna(row.get('Что нужно для занятий')) else None,
            str(row.get('Чему ребенок научится/ что изучит?', '')).strip() if pd.notna(row.get('Чему ребенок научится/ что изучит?')) else None,
            str(row.get('Педагог', '')).strip() if pd.notna(row.get('Педагог')) else None,
            str(row.get('Информация о педагоге', '')).strip() if pd.notna(row.get('Информация о педагоге')) else None,
            str(row.get('Ссылка на запись (навигатор)', '')).strip() if pd.notna(row.get('Ссылка на запись (навигатор)')) else None,
            chunk,
            search,
        ))

        program_id = cur.lastrowid

        # Генерируем теги из tag_synonyms (по направлению и названию)
        direction = str(row.get('Направление', '')).lower()
        name = str(row.get('Название коллектива', '')).lower()
        combined = direction + ' ' + name

        tags_added = set()
        for tag_group, synonyms in TAG_SYNONYMS.items():
            for syn in synonyms:
                if syn in combined:
                    if tag_group not in tags_added:
                        cur.execute('INSERT INTO program_tags (program_id, tag) VALUES (?, ?)',
                                    (program_id, tag_group))
                        tags_added.add(tag_group)
                    break

        # Добавляем теги из колонки 'теги' в Excel (напрямую)
        excel_tags = row.get('теги')
        if pd.notna(excel_tags):
            for tag in str(excel_tags).split(','):
                tag = tag.strip().lower()
                if tag and tag not in tags_added:
                    cur.execute('INSERT INTO program_tags (program_id, tag) VALUES (?, ?)',
                                (program_id, tag))
                    tags_added.add(tag)

        # Добавляем тег стоимости
        if is_free and 'бюджет' not in tags_added:
            cur.execute('INSERT INTO program_tags (program_id, tag) VALUES (?, ?)',
                        (program_id, 'бюджет'))
        elif not is_free and 'платно' not in tags_added:
            cur.execute('INSERT INTO program_tags (program_id, tag) VALUES (?, ?)',
                        (program_id, 'платно'))

    # ─── Загружаем FAQ ───
    faq_items = load_faq(faq_path)
    print(f"Загружено {len(faq_items)} FAQ вопросов")

    for item in faq_items:
        search = f"{item['question']} {item['answer']}"
        cur.execute('''
            INSERT INTO faq (question, answer, search_text)
            VALUES (?, ?, ?)
        ''', (item['question'], item['answer'], search))

    # ─── FTS5 индексы ───
    cur.execute('''
        CREATE VIRTUAL TABLE programs_fts USING fts5(
            name, direction, search_text,
            content='programs',
            content_rowid='id',
            tokenize='unicode61'
        )
    ''')

    cur.execute('''
        INSERT INTO programs_fts (rowid, name, direction, search_text)
        SELECT id, name, direction, search_text FROM programs
    ''')

    cur.execute('''
        CREATE VIRTUAL TABLE faq_fts USING fts5(
            search_text,
            content='faq',
            content_rowid='id',
            tokenize='unicode61'
        )
    ''')

    cur.execute('''
        INSERT INTO faq_fts (rowid, search_text)
        SELECT id, search_text FROM faq
    ''')

    conn.commit()

    # Статистика
    cur.execute('SELECT COUNT(*) FROM programs')
    n_programs = cur.fetchone()[0]
    cur.execute('SELECT COUNT(*) FROM faq')
    n_faq = cur.fetchone()[0]
    cur.execute('SELECT COUNT(*) FROM program_tags')
    n_tags = cur.fetchone()[0]

    print(f"\n✅ База данных создана: {db_path}")
    print(f"   Программ: {n_programs}")
    print(f"   FAQ: {n_faq}")
    print(f"   Тегов: {n_tags}")

    conn.close()
    return db_path


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', '..', 'data')

    programs = sys.argv[1] if len(sys.argv) > 1 else os.path.join(data_dir, 'programs.xlsx')
    faq = sys.argv[2] if len(sys.argv) > 2 else os.path.join(data_dir, 'FAQ.docx')
    db_out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(script_dir, 'knowledge.db')
    create_database(programs, faq, db_out)