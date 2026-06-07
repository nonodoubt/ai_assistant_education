"""
scripts/validate_programs.py — Валидация programs.xlsx перед сборкой БД.

Запускается клиентом или разработчиком после редактирования файла.
Проверяет структуру, типы данных, допустимые значения и предупреждает
об ошибках, которые сломают build_db.py или приведут к потере данных.

Использование:
    python scripts/validate_programs.py                         # проверяет data/programs.xlsx
    python scripts/validate_programs.py path/to/programs.xlsx   # произвольный путь

Уровни сообщений:
    ❌ ОШИБКА   — БД не соберётся или данные потеряются. Нужно исправить.
    ⚠️  ВНИМАНИЕ — работать будет, но результат может быть неточным.
    ℹ️  ИНФО     — рекомендация по улучшению качества данных.
"""

import os
import re
import sys

try:
    import pandas as pd
except ImportError:
    print("Нужен pandas: pip install pandas openpyxl")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════
# ЭТАЛОН: обязательные колонки и их порядок (из build_db.py)
# ═══════════════════════════════════════════════════════════════════

REQUIRED_COLUMNS = [
    'Направление',
    'Название коллектива',
    'Возраст',
    'Платное/Бюджет',
    'Расписание',
    'Площадка',
    'Что нужно для занятий',
    'Чему ребенок научится/ что изучит?',
    'Педагог',
    'Информация о педагоге',
    'Ссылка на запись (навигатор)',
    'теги',
]

# Допустимые значения для категориальных полей
VALID_PRICE = {'Бюджет', '500 рублей / занятие'}
VALID_LOCATION = {'пр. Раевского 5/2', 'пр Тореза 35/1'}

# Паттерн возраста: "N-M лет/года" или "N лет/года"
AGE_PATTERN = re.compile(
    r'^\d{1,2}\s*[-–]\s*\d{1,2}\s*(лет|года?|г\.?)$'
    r'|'
    r'^\d{1,2}\s*(лет|года?|г\.?)$',
    re.IGNORECASE,
)

# URL-паттерн (простой)
URL_PATTERN = re.compile(r'^https?://', re.IGNORECASE)


def validate(filepath):
    errors = []    # ❌
    warnings = []  # ⚠️
    infos = []     # ℹ️

    def err(msg):   errors.append(msg)
    def warn(msg):  warnings.append(msg)
    def info(msg):  infos.append(msg)

    # ─── 1. Чтение файла ───
    try:
        df = pd.read_excel(filepath)
    except FileNotFoundError:
        err(f"Файл не найден: {filepath}")
        return errors, warnings, infos
    except Exception as e:
        err(f"Не удалось прочитать файл: {e}")
        return errors, warnings, infos

    if df.empty:
        err("Файл пуст — нет ни одной строки данных.")
        return errors, warnings, infos

    # ─── 2. Проверка колонок ───
    actual_cols = list(df.columns)

    # Убираем Unnamed колонки (Excel иногда добавляет пустые)
    unnamed = [c for c in actual_cols if str(c).startswith('Unnamed')]
    if unnamed:
        warn(f"Обнаружены пустые колонки без заголовка: {unnamed}. "
             f"Они будут проигнорированы. Удалите лишние столбцы в Excel.")
        df = df.drop(columns=unnamed)
        actual_cols = list(df.columns)

    # Недостающие обязательные колонки
    missing = [c for c in REQUIRED_COLUMNS if c not in actual_cols]
    if missing:
        err(f"Отсутствуют обязательные колонки: {missing}")
        # Показываем что есть, чтобы помочь найти опечатку
        for m in missing:
            # Ищем похожие (без учёта регистра и пробелов)
            m_norm = m.lower().strip()
            for a in actual_cols:
                a_norm = a.lower().strip()
                if m_norm in a_norm or a_norm in m_norm:
                    info(f"  Возможно, '{a}' — это '{m}' (опечатка / лишний пробел)?")

    # Лишние колонки
    extra = [c for c in actual_cols if c not in REQUIRED_COLUMNS]
    if extra:
        warn(f"Обнаружены нестандартные колонки: {extra}. "
             f"Они не сломают сборку, но и не будут использоваться ботом.")

    # Порядок колонок (не критично, но может указывать на проблему)
    expected_order = [c for c in REQUIRED_COLUMNS if c in actual_cols]
    actual_order = [c for c in actual_cols if c in REQUIRED_COLUMNS]
    if expected_order != actual_order:
        info("Порядок колонок отличается от эталонного. Это не ошибка, "
             "но может запутать при ручном редактировании.")

    # Если не хватает ключевых колонок — дальше проверять нет смысла
    critical_missing = {'Направление', 'Название коллектива', 'Возраст',
                        'Платное/Бюджет', 'Площадка'} & set(missing)
    if critical_missing:
        err(f"Критические колонки отсутствуют: {critical_missing}. "
            f"Дальнейшая валидация невозможна.")
        return errors, warnings, infos

    # ─── 3. Проверка данных по строкам ───
    for idx, row in df.iterrows():
        row_num = idx + 2  # Excel: строка 1 = заголовок, данные с 2
        prefix = f"Строка {row_num}"

        # ── Направление ──
        direction = row.get('Направление')
        if pd.isna(direction) or not str(direction).strip():
            err(f"{prefix}: пустое Направление.")
        else:
            d = str(direction)
            if d != d.strip():
                warn(f"{prefix}: Направление '{d}' содержит лишние пробелы "
                     f"в начале или конце. Это может привести к дублям "
                     f"в фильтре админки.")

        # ── Название коллектива ──
        name = row.get('Название коллектива')
        if pd.isna(name) or not str(name).strip():
            err(f"{prefix}: пустое Название коллектива.")

        # ── Возраст ──
        age = row.get('Возраст')
        if pd.isna(age) or not str(age).strip():
            err(f"{prefix}: пустой Возраст.")
        else:
            age_s = str(age).strip()
            if not AGE_PATTERN.match(age_s):
                warn(f"{prefix}: Возраст '{age_s}' не соответствует формату "
                     f"'N-M лет' или 'N лет'. Бот может некорректно "
                     f"фильтровать по возрасту.")
            else:
                nums = re.findall(r'\d+', age_s)
                if len(nums) >= 2 and int(nums[0]) > int(nums[1]):
                    err(f"{prefix}: в Возрасте '{age_s}' минимум ({nums[0]}) "
                        f"больше максимума ({nums[1]}).")
                if nums and int(nums[0]) > 18:
                    warn(f"{prefix}: Возраст '{age_s}' — начало > 18 лет. "
                         f"Это детский центр, проверьте данные.")

        # ── Платное/Бюджет ──
        price = row.get('Платное/Бюджет')
        if pd.isna(price) or not str(price).strip():
            warn(f"{prefix}: пустое поле Платное/Бюджет. Бот не сможет "
                 f"определить стоимость.")
        else:
            p = str(price).strip()
            if p not in VALID_PRICE:
                warn(f"{prefix}: Платное/Бюджет = '{p}' — нестандартное значение. "
                     f"Допустимые: {VALID_PRICE}. Бот может некорректно "
                     f"определять стоимость.")

        # ── Площадка ──
        location = row.get('Площадка')
        if pd.isna(location) or not str(location).strip():
            err(f"{prefix}: пустая Площадка.")
        else:
            loc = str(location).strip()
            if loc not in VALID_LOCATION:
                # Проверяем хотя бы по ключевым словам
                loc_lower = loc.lower()
                if 'раевск' not in loc_lower and 'торез' not in loc_lower:
                    warn(f"{prefix}: Площадка = '{loc}' — не распознана. "
                         f"Допустимые: {VALID_LOCATION}.")
                else:
                    info(f"{prefix}: Площадка = '{loc}' — похожа на допустимую, "
                         f"но написана иначе. Рекомендуется привести к одному "
                         f"из: {VALID_LOCATION}.")

        # ── Расписание ──
        schedule = row.get('Расписание')
        if pd.isna(schedule) or not str(schedule).strip():
            warn(f"{prefix}: пустое Расписание. Бот не сможет "
                 f"информировать родителей о днях занятий.")

        # ── Педагог ──
        teacher = row.get('Педагог')
        if pd.isna(teacher) or not str(teacher).strip():
            info(f"{prefix}: не указан Педагог.")

        # ── Ссылка на запись ──
        url = row.get('Ссылка на запись (навигатор)')
        if pd.notna(url) and str(url).strip():
            u = str(url).strip()
            if not URL_PATTERN.match(u):
                warn(f"{prefix}: Ссылка на запись '{u[:60]}...' "
                     f"не похожа на URL (нет http:// или https://).")

        # ── Теги ──
        tags = row.get('теги')
        if pd.isna(tags) or not str(tags).strip():
            info(f"{prefix}: пустые теги. Поиск по синонимам будет "
                 f"работать хуже для этой программы.")

    # ─── 4. Проверка консистентности ───

    # Дубли направлений с пробелами (частая ошибка)
    if 'Направление' in df.columns:
        dirs = df['Направление'].dropna().apply(str)
        stripped = dirs.str.strip()
        diff_mask = dirs != stripped
        if diff_mask.any():
            bad_rows = list(df[diff_mask].index + 2)
            warn(f"В колонке 'Направление' есть значения с лишними пробелами "
                 f"(строки: {bad_rows}). Это создаёт дубли в админке: "
                 f"например 'Шахматы' и 'Шахматы ' считаются разными.")

        # Группируем по stripped и смотрим оригиналы
        unique_raw = dirs.unique()
        unique_stripped = set(s.strip() for s in unique_raw)
        if len(unique_raw) != len(unique_stripped):
            groups = {}
            for d in unique_raw:
                groups.setdefault(d.strip(), []).append(repr(d))
            for k, variants in groups.items():
                if len(variants) > 1:
                    warn(f"Направление '{k}' записано по-разному: "
                         f"{', '.join(variants)}. Приведите к единому виду.")

    # Площадки
    if 'Площадка' in df.columns:
        locs = df['Площадка'].dropna().apply(lambda x: str(x).strip()).unique()
        if len(locs) > 2:
            warn(f"Обнаружено {len(locs)} уникальных площадок: {list(locs)}. "
                 f"У ДДТ «Союз» две площадки. Проверьте, нет ли опечаток.")

    # Полные дубликаты строк
    if 'Название коллектива' in df.columns and 'Возраст' in df.columns:
        dup_cols = ['Название коллектива', 'Возраст', 'Площадка']
        dup_cols = [c for c in dup_cols if c in df.columns]
        dups = df[df.duplicated(subset=dup_cols, keep=False)]
        if not dups.empty:
            dup_rows = list(dups.index + 2)
            warn(f"Возможные дубликаты программ (одинаковое название + "
                 f"возраст + площадка), строки: {dup_rows}. "
                 f"Проверьте, не добавлена ли программа дважды.")

    # Пустые строки (все ключевые поля пустые)
    key_cols = ['Направление', 'Название коллектива', 'Возраст']
    key_cols = [c for c in key_cols if c in df.columns]
    if key_cols:
        empty_mask = df[key_cols].apply(
            lambda row: all(pd.isna(v) or str(v).strip() == '' for v in row),
            axis=1
        )
        if empty_mask.any():
            empty_rows = list(df[empty_mask].index + 2)
            warn(f"Полностью пустые строки: {empty_rows}. Удалите их из файла.")

    return errors, warnings, infos


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', '..', 'data')
    filepath = sys.argv[1] if len(sys.argv) > 1 else os.path.join(data_dir, 'programs.xlsx')

    print("=" * 60)
    print("  Валидация programs.xlsx")
    print("=" * 60)
    print(f"  Файл: {filepath}")
    print()

    errors, warnings, infos = validate(filepath)

    if errors:
        print(f"❌ ОШИБКИ ({len(errors)}):")
        for e in errors:
            print(f"   ❌ {e}")
        print()

    if warnings:
        print(f"⚠️  ВНИМАНИЕ ({len(warnings)}):")
        for w in warnings:
            print(f"   ⚠️  {w}")
        print()

    if infos:
        print(f"ℹ️  ИНФО ({len(infos)}):")
        for i in infos:
            print(f"   ℹ️  {i}")
        print()

    # Итог
    print("-" * 60)
    if errors:
        print(f"🛑 Найдено {len(errors)} ошибок — НЕЛЬЗЯ собирать БД.")
        print("   Исправьте ошибки и запустите проверку заново.")
    elif warnings:
        print(f"✅ Ошибок нет, но {len(warnings)} предупреждений.")
        print("   БД соберётся, но рекомендуется проверить предупреждения.")
    else:
        print("✅ Всё в порядке! Можно собирать БД:")
        print("   python services/db/build_db.py")
        print("   python services/db/build_vectors.py")

    sys.exit(1 if errors else 0)


if __name__ == '__main__':
    main()