"""
services/db/build_vectors.py — Генерация эмбеддингов и запись в sqlite-vec.

Модель и размерность берутся из config/prompts.py (EMBEDDING_MODEL, EMBEDDING_DIM)
— единая точка для всего проекта. Текущая модель: intfloat/multilingual-e5-base
(мультиязычная, 768 измерений, поддерживает русский).
Хранение: sqlite-vec расширение для SQLite.

Запуск:
    python services/db/build_vectors.py        (из корня проекта)
    # или:  cd services/db && python3 build_vectors.py

Зависимости:
    pip install sqlite-vec sentence-transformers torch
"""

import sqlite3
import os
import sys
import json
import struct
import numpy as np

# ─── Конфиг модели (единая точка) ───
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
try:
    from config.prompts import EMBEDDING_MODEL, EMBEDDING_DIM
except Exception:
    EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
    EMBEDDING_DIM = 768

DB_NAME = "knowledge.db"


def serialize_f32(vector):
    """Сериализует numpy/list вектор в bytes для sqlite-vec."""
    return struct.pack("%sf" % len(vector), *vector)


def build_embedding_text(row):
    """
    Формирует текст для эмбеддинга программы.
    Включает ВСЁ что может искать пользователь.
    """
    parts = []
    if row.get('name'):
        parts.append(row['name'])
    if row.get('direction'):
        parts.append(row['direction'])
    if row.get('teacher'):
        parts.append("Педагог: %s" % row['teacher'])
    if row.get('age_str'):
        parts.append("Возраст: %s" % row['age_str'])
    if row.get('results'):
        parts.append(row['results'])
    if row.get('requirements'):
        parts.append(row['requirements'])
    if row.get('schedule'):
        parts.append(row['schedule'])
    # Теги из search_text (содержат теги из Excel)
    if row.get('search_text'):
        parts.append(row['search_text'])
    return '. '.join(parts)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, DB_NAME)

    if not os.path.exists(db_path):
        print("❌ БД не найдена: %s" % db_path)
        print("   Сначала запустите: python3 build_db.py")
        sys.exit(1)

    print("=" * 60)
    print("  Генерация эмбеддингов для RAG-бота ДДТ «Союз»")
    print("=" * 60)

    # ─── Загрузка модели ───
    print("\n1. Загрузка модели %s..." % EMBEDDING_MODEL)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("   ✅ Модель загружена (dim=%d)" % EMBEDDING_DIM)

    # ─── Подключение к БД + sqlite-vec ───
    print("\n2. Подключение к БД...")
    import sqlite_vec
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    cur = conn.cursor()

    # Удаляем старую таблицу если есть
    cur.execute("DROP TABLE IF EXISTS vec_programs")
    cur.execute("DROP TABLE IF EXISTS vec_faq")

    # Создаём таблицы для векторов
    cur.execute("""
        CREATE VIRTUAL TABLE vec_programs USING vec0(
            embedding float[%d]
        )
    """ % EMBEDDING_DIM)

    cur.execute("""
        CREATE VIRTUAL TABLE vec_faq USING vec0(
            embedding float[%d]
        )
    """ % EMBEDDING_DIM)

    # ─── Эмбеддинги программ ───
    print("\n3. Генерация эмбеддингов программ...")
    cur.execute('SELECT * FROM programs')
    columns = [desc[0] for desc in cur.description]
    programs = [dict(zip(columns, row)) for row in cur.fetchall()]

    texts = []
    ids = []
    for p in programs:
        text = "passage: " + build_embedding_text(p)
        texts.append(text)
        ids.append(p['id'])

    # Batch encode
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    for prog_id, emb in zip(ids, embeddings):
        cur.execute(
            "INSERT INTO vec_programs(rowid, embedding) VALUES (?, ?)",
            (prog_id, serialize_f32(emb))
        )

    print("   ✅ %d эмбеддингов программ" % len(ids))

    # ─── Эмбеддинги FAQ ───
    print("\n4. Генерация эмбеддингов FAQ...")
    cur.execute('SELECT id, question, answer, search_text FROM faq')
    faqs = cur.fetchall()

    faq_texts = []
    faq_ids = []
    for faq in faqs:
        text = "passage: %s %s" % (faq[1], faq[2])
        faq_texts.append(text)
        faq_ids.append(faq[0])

    faq_embeddings = model.encode(faq_texts, normalize_embeddings=True)

    for faq_id, emb in zip(faq_ids, faq_embeddings):
        cur.execute(
            "INSERT INTO vec_faq(rowid, embedding) VALUES (?, ?)",
            (faq_id, serialize_f32(emb))
        )

    print("   ✅ %d эмбеддингов FAQ" % len(faq_ids))

    # ─── Сохранение ───
    conn.commit()

    # Статистика
    cur.execute("SELECT COUNT(*) FROM vec_programs")
    n_vec_prog = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM vec_faq")
    n_vec_faq = cur.fetchone()[0]

    print("\n" + "=" * 60)
    print("  ✅ Готово!")
    print("  Векторов программ: %d" % n_vec_prog)
    print("  Векторов FAQ: %d" % n_vec_faq)
    print("  Модель: %s" % EMBEDDING_MODEL)
    print("  Размерность: %d" % EMBEDDING_DIM)
    print("  БД: %s" % db_path)
    print("=" * 60)

    conn.close()


if __name__ == "__main__":
    main()