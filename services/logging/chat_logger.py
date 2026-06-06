"""
services/logging/chat_logger.py

Логирование чатов в SQLite:
- sessions: одна запись на сессию (Streamlit запуск пользователя)
- messages: каждое сообщение (user/assistant)
- feedback: лайки/дизлайки на ответы бота

БД: data/chat_logs.db (отдельно от knowledge.db)
"""

import sqlite3
import json
import os
import time
import uuid
from datetime import datetime
from typing import Optional, Dict, Any


# ═══════════════════════════════════════════════════════════════════════════════
# СХЕМА
# ═══════════════════════════════════════════════════════════════════════════════

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    started_at TIMESTAMP NOT NULL,
    last_active TIMESTAMP NOT NULL,
    user_agent TEXT,
    message_count INTEGER DEFAULT 0,
    feedback_likes INTEGER DEFAULT 0,
    feedback_dislikes INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    llm_calls INTEGER DEFAULT 0,
    model_used TEXT,
    response_time_seconds REAL,
    keywords TEXT,
    age_collected INTEGER,
    feedback TEXT DEFAULT NULL,
    feedback_timestamp TIMESTAMP DEFAULT NULL,
    served_directions TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS surveys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    q1_satisfaction INTEGER,
    q2_convenient TEXT,
    q3_recommend INTEGER,
    q4_speed INTEGER,
    q5_relevance INTEGER,
    comment TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_feedback ON messages(feedback);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at);
"""


class ChatLogger:
    """Логирует сессии, сообщения и фидбек в SQLite."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.executescript(SCHEMA)
        # Миграция: для существующих БД, где колонки served_directions ещё нет.
        # SQLite ALTER не поддерживает IF NOT EXISTS, поэтому проверяем явно.
        cols = {row[1] for row in conn.execute("PRAGMA table_info(messages)")}
        if 'served_directions' not in cols:
            conn.execute("ALTER TABLE messages ADD COLUMN served_directions TEXT")
            print("[chat_logger] миграция: добавлена колонка messages.served_directions")
        conn.commit()
        conn.close()

    def _conn(self):
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c

    # ─── Сессии ───

    def create_session(self, user_agent: Optional[str] = None) -> str:
        """Создаёт новую сессию. Возвращает session_id (UUID)."""
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        with self._conn() as c:
            c.execute("""
                INSERT INTO sessions (session_id, started_at, last_active, user_agent)
                VALUES (?, ?, ?, ?)
            """, (session_id, now, now, user_agent or ""))
            c.commit()
        return session_id

    def touch_session(self, session_id: str):
        """Обновляет last_active."""
        with self._conn() as c:
            c.execute("UPDATE sessions SET last_active = ? WHERE session_id = ?",
                      (datetime.now().isoformat(), session_id))
            c.commit()

    # ─── Сообщения ───

    def log_message(self, session_id: str, role: str, content: str,
                    llm_calls: int = 0, model_used: Optional[str] = None,
                    response_time_seconds: Optional[float] = None,
                    keywords: Optional[list] = None,
                    age_collected: Optional[int] = None,
                    served_directions: Optional[list] = None) -> int:
        """
        Логирует одно сообщение. Возвращает id записи (для последующего фидбека).

        served_directions: для assistant-сообщений — список направлений тех программ,
        которые бот реально включил в ответ. Для FAQ/шаблонов/уточнений — None или [].
        """
        kw_json = json.dumps(keywords, ensure_ascii=False) if keywords else None
        sd_norm = None
        if served_directions:
            # Нормализуем: strip, без дубликатов, без пустых
            sd_clean = sorted({d.strip() for d in served_directions if d and d.strip()})
            if sd_clean:
                sd_norm = json.dumps(sd_clean, ensure_ascii=False)
        with self._conn() as c:
            cur = c.execute("""
                INSERT INTO messages (
                    session_id, role, content, timestamp,
                    llm_calls, model_used, response_time_seconds, keywords, age_collected,
                    served_directions
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id, role, content, datetime.now().isoformat(),
                llm_calls, model_used, response_time_seconds, kw_json, age_collected,
                sd_norm,
            ))
            c.execute("UPDATE sessions SET message_count = message_count + 1, last_active = ? "
                      "WHERE session_id = ?",
                      (datetime.now().isoformat(), session_id))
            c.commit()
            return cur.lastrowid

    # ─── Фидбек ───

    def set_feedback(self, message_id: int, feedback: str) -> bool:
        """
        feedback: 'like' / 'dislike' / None (отменить).
        Возвращает True если обновлено.
        """
        if feedback not in ('like', 'dislike', None):
            return False
        with self._conn() as c:
            # Получаем текущий feedback и session_id
            row = c.execute(
                "SELECT feedback, session_id FROM messages WHERE id = ?",
                (message_id,)
            ).fetchone()
            if not row:
                return False
            old_feedback = row['feedback']
            session_id = row['session_id']

            # Обновляем
            ts = datetime.now().isoformat() if feedback else None
            c.execute(
                "UPDATE messages SET feedback = ?, feedback_timestamp = ? WHERE id = ?",
                (feedback, ts, message_id)
            )

            # Обновляем счётчики в session
            delta_like = 0
            delta_dislike = 0
            if old_feedback == 'like':
                delta_like -= 1
            elif old_feedback == 'dislike':
                delta_dislike -= 1
            if feedback == 'like':
                delta_like += 1
            elif feedback == 'dislike':
                delta_dislike += 1

            if delta_like or delta_dislike:
                c.execute("""
                    UPDATE sessions
                    SET feedback_likes = feedback_likes + ?,
                        feedback_dislikes = feedback_dislikes + ?
                    WHERE session_id = ?
                """, (delta_like, delta_dislike, session_id))

            c.commit()
            return True

    # ─── Аналитика ───

    def get_session_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if not row:
                return None
            return dict(row)

    def get_recent_sessions(self, limit: int = 20) -> list:
        with self._conn() as c:
            rows = c.execute("""
                SELECT * FROM sessions
                ORDER BY last_active DESC
                LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in rows]

    # ─── Фильтры + пагинация + направления ───
    #
    # Источник истины для направлений — колонка messages.served_directions.
    # Она заполняется ботом ТОЛЬКО когда в ответе реально были программы;
    # FAQ-ответы и шаблоны направлений не имеют (NULL). Сессия попадает в
    # фильтр «по направлению X», если хотя бы одно её assistant-сообщение
    # содержит X в served_directions.

    def _session_directions(self, messages) -> set:
        """Собирает множество направлений, реально выданных в сессии."""
        all_dirs = set()
        for m in messages:
            sd_raw = m.get('served_directions')
            if not sd_raw:
                continue
            try:
                sd = json.loads(sd_raw)
            except Exception:
                continue
            if isinstance(sd, list):
                all_dirs.update(d.strip() for d in sd if d and d.strip())
        return all_dirs

    def get_sessions_filtered(self, *, page=1, page_size=15,
                              date_from=None, date_to=None,
                              duration_min=None, duration_max=None,
                              user_msg_min=None, user_msg_max=None,
                              likes_min=None, likes_max=None,
                              dislikes_min=None, dislikes_max=None,
                              directions=None) -> dict:
        """
        Возвращает {sessions, total, page, page_size, pages}.
        Скалярные фильтры выполняются в SQL.
        Фильтр по направлениям применяется на уровне сессий:
        сессия проходит, если хотя бы одно её assistant-сообщение содержит
        хотя бы одно из запрошенных направлений в served_directions.
        """
        clauses = ["1=1"]
        params = []
        if date_from:
            clauses.append("s.started_at >= ?")
            params.append(date_from)
        if date_to:
            clauses.append("s.started_at <= ?")
            params.append(date_to + " 23:59:59")
        if duration_min is not None:
            clauses.append(
                "(julianday(s.last_active) - julianday(s.started_at)) * 86400.0 >= ?")
            params.append(float(duration_min))
        if duration_max is not None:
            clauses.append(
                "(julianday(s.last_active) - julianday(s.started_at)) * 86400.0 <= ?")
            params.append(float(duration_max))
        if likes_min is not None:
            clauses.append("s.feedback_likes >= ?")
            params.append(int(likes_min))
        if likes_max is not None:
            clauses.append("s.feedback_likes <= ?")
            params.append(int(likes_max))
        if dislikes_min is not None:
            clauses.append("s.feedback_dislikes >= ?")
            params.append(int(dislikes_min))
        if dislikes_max is not None:
            clauses.append("s.feedback_dislikes <= ?")
            params.append(int(dislikes_max))
        umsg_select = ("(SELECT COUNT(*) FROM messages m WHERE m.session_id = "
                       "s.session_id AND m.role = 'user') AS user_msg_count")
        having = []
        if user_msg_min is not None:
            having.append("user_msg_count >= %d" % int(user_msg_min))
        if user_msg_max is not None:
            having.append("user_msg_count <= %d" % int(user_msg_max))

        sql_where = " AND ".join(clauses)
        sql_having = (" HAVING " + " AND ".join(having)) if having else ""
        sql = ("SELECT s.*, "
               "(julianday(s.last_active) - julianday(s.started_at)) * 86400.0 "
               "AS duration_seconds, " + umsg_select +
               " FROM sessions s WHERE " + sql_where +
               " GROUP BY s.session_id" + sql_having +
               " ORDER BY s.last_active DESC")

        with self._conn() as c:
            rows = c.execute(sql, params).fetchall()
            candidates = [dict(r) for r in rows]
            wanted = set(d.strip() for d in directions) if directions else None
            enriched = []
            for s in candidates:
                msgs = c.execute(
                    "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC",
                    (s['session_id'],)
                ).fetchall()
                msgs = [dict(m) for m in msgs]
                s['messages'] = msgs
                s['directions'] = sorted(self._session_directions(msgs))
                if wanted and not (set(s['directions']) & wanted):
                    continue
                enriched.append(s)

        total = len(enriched)
        pages = max(1, (total + page_size - 1) // page_size)
        page = max(1, min(int(page), pages))
        start = (page - 1) * page_size
        return {
            'sessions': enriched[start:start + page_size],
            'total': total, 'page': page, 'page_size': page_size, 'pages': pages,
        }

    def get_directions_summary(self, *, date_from=None, date_to=None) -> dict:
        """
        Топ направлений по реальным выдачам.
        Возвращает {summary, all_directions}:
          summary: [{direction, sessions, messages}]  отсортировано по sessions desc
          all_directions: отсортированный список всех встречавшихся направлений
                         (для UI-фильтра)
        Считаются ТОЛЬКО сообщения, где served_directions не NULL — т.е.
        реальные выдачи программ. FAQ и шаблоны в статистику не идут.
        """
        clauses = ["s.session_id IS NOT NULL"]
        params = []
        if date_from:
            clauses.append("s.started_at >= ?")
            params.append(date_from)
        if date_to:
            clauses.append("s.started_at <= ?")
            params.append(date_to + " 23:59:59")
        sql = ("SELECT m.session_id, m.served_directions "
               "FROM messages m JOIN sessions s ON s.session_id = m.session_id "
               "WHERE m.served_directions IS NOT NULL AND " +
               " AND ".join(clauses))
        sess_count = {}
        msg_count = {}
        seen_sess_for_dir = {}
        with self._conn() as c:
            for row in c.execute(sql, params).fetchall():
                sid = row['session_id']
                try:
                    dirs = json.loads(row['served_directions'])
                except Exception:
                    continue
                if not isinstance(dirs, list):
                    continue
                for d in dirs:
                    d = d.strip()
                    if not d:
                        continue
                    msg_count[d] = msg_count.get(d, 0) + 1
                    s_set = seen_sess_for_dir.setdefault(d, set())
                    if sid not in s_set:
                        s_set.add(sid)
                        sess_count[d] = sess_count.get(d, 0) + 1
        summary = [
            {'direction': d, 'sessions': sess_count[d],
             'messages': msg_count.get(d, 0)}
            for d in sess_count
        ]
        summary.sort(key=lambda x: (-x['sessions'], -x['messages']))
        return {
            'summary': summary,
            'all_directions': sorted(sess_count.keys()),
        }

    def get_session_messages(self, session_id: str) -> list:
        with self._conn() as c:
            rows = c.execute("""
                SELECT * FROM messages
                WHERE session_id = ?
                ORDER BY id ASC
            """, (session_id,)).fetchall()
            return [dict(r) for r in rows]

    def get_disliked_messages(self, limit: int = 50) -> list:
        """Все ответы с дизлайком + предыдущее user-сообщение для контекста."""
        with self._conn() as c:
            rows = c.execute("""
                SELECT * FROM messages
                WHERE feedback = 'dislike'
                ORDER BY feedback_timestamp DESC
                LIMIT ?
            """, (limit,)).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                # Предыдущее user-сообщение
                prev = c.execute("""
                    SELECT content FROM messages
                    WHERE session_id = ? AND id < ? AND role = 'user'
                    ORDER BY id DESC LIMIT 1
                """, (d['session_id'], d['id'])).fetchone()
                d['user_question'] = prev['content'] if prev else ""
                result.append(d)
            return result

    def get_liked_messages(self, limit: int = 50) -> list:
        with self._conn() as c:
            rows = c.execute("""
                SELECT * FROM messages
                WHERE feedback = 'like'
                ORDER BY feedback_timestamp DESC
                LIMIT ?
            """, (limit,)).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                prev = c.execute("""
                    SELECT content FROM messages
                    WHERE session_id = ? AND id < ? AND role = 'user'
                    ORDER BY id DESC LIMIT 1
                """, (d['session_id'], d['id'])).fetchone()
                d['user_question'] = prev['content'] if prev else ""
                result.append(d)
            return result

    def get_summary_stats(self):
        """Общая статистика."""
        with self._conn() as c:
            sessions = c.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            messages = c.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            user_msg = c.execute("SELECT COUNT(*) FROM messages WHERE role='user'").fetchone()[0]
            assistant_msg = c.execute(
                "SELECT COUNT(*) FROM messages WHERE role='assistant'"
            ).fetchone()[0]
            likes = c.execute(
                "SELECT COUNT(*) FROM messages WHERE feedback='like'"
            ).fetchone()[0]
            dislikes = c.execute(
                "SELECT COUNT(*) FROM messages WHERE feedback='dislike'"
            ).fetchone()[0]
            avg_msgs = c.execute(
                "SELECT AVG(message_count) FROM sessions WHERE message_count > 0"
            ).fetchone()[0] or 0
            surveys = c.execute("SELECT COUNT(*) FROM surveys").fetchone()[0]
            return {
                'total_sessions': sessions,
                'total_messages': messages,
                'user_messages': user_msg,
                'assistant_messages': assistant_msg,
                'likes': likes,
                'dislikes': dislikes,
                'avg_messages_per_session': round(avg_msgs, 1),
                'surveys': surveys,
            }

    # ─── Опросы ───

    def save_survey(self, session_id, q1, q2, q3, q4, q5, comment=None):
        """Сохраняет результат опроса обратной связи."""
        with self._conn() as c:
            c.execute("""
                INSERT INTO surveys (session_id, timestamp, q1_satisfaction,
                    q2_convenient, q3_recommend, q4_speed, q5_relevance, comment)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (session_id, datetime.now().isoformat(), q1, q2, q3, q4, q5, comment))
            c.commit()

    def get_surveys(self, limit=50):
        with self._conn() as c:
            rows = c.execute("""
                SELECT s.*, ss.started_at, ss.message_count
                FROM surveys s
                LEFT JOIN sessions ss ON s.session_id = ss.session_id
                ORDER BY s.timestamp DESC LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in rows]

    def get_survey_averages(self):
        with self._conn() as c:
            row = c.execute("""
                SELECT COUNT(*) as cnt,
                       AVG(q1_satisfaction) as avg_q1,
                       AVG(q3_recommend) as avg_q3,
                       AVG(q4_speed) as avg_q4,
                       AVG(q5_relevance) as avg_q5,
                       SUM(CASE WHEN q2_convenient='Да' THEN 1 ELSE 0 END) as q2_yes,
                       SUM(CASE WHEN q2_convenient='Нет' THEN 1 ELSE 0 END) as q2_no
                FROM surveys
            """).fetchone()
            if not row or row['cnt'] == 0:
                return None
            return dict(row)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI просмотр статистики
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import sys
    db_path = sys.argv[1] if len(sys.argv) > 1 else None
    if db_path is None:
        # По умолчанию ищем data/chat_logs.db от корня проекта
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for path in [
            os.path.join(script_dir, '..', '..', 'data', 'chat_logs.db'),
            os.path.join(script_dir, '..', 'chat_logs.db'),
        ]:
            if os.path.exists(path):
                db_path = path
                break

    if not db_path or not os.path.exists(db_path):
        print("БД логов не найдена")
        return

    logger = ChatLogger(db_path)
    stats = logger.get_summary_stats()

    print("=" * 70)
    print("  Статистика чат-бота")
    print("=" * 70)
    print(f"  Сессий:                    {stats['total_sessions']}")
    print(f"  Сообщений всего:           {stats['total_messages']}")
    print(f"   - от пользователей:       {stats['user_messages']}")
    print(f"   - от бота:                {stats['assistant_messages']}")
    print(f"  Лайков:                    {stats['likes']}")
    print(f"  Дизлайков:                 {stats['dislikes']}")
    print(f"  Среднее сообщений/сессия:  {stats['avg_messages_per_session']}")

    if stats['dislikes'] > 0:
        print("\n" + "─" * 70)
        print("  Последние ответы с дизлайком:")
        print("─" * 70)
        for d in logger.get_disliked_messages(limit=10):
            print(f"\n  [{d.get('feedback_timestamp', '?')[:19]}]")
            print(f"  Q: {d.get('user_question', '')[:100]}")
            print(f"  A: {d['content'][:150]}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()