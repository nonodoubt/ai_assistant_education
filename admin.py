"""
admin.py — Админ-панель для просмотра логов чат-бота.

Запуск:
    streamlit run admin.py --server.port 8502

Показывает:
- Общую статистику (сессии, сообщения, лайки/дизлайки)
- Список сессий с возможностью провалиться в каждую
- Список ответов с дизлайками (для анализа проблемных запросов)
- Список ответов с лайками (что работает хорошо)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from datetime import datetime
import json

from services.logging.chat_logger import ChatLogger

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHAT_LOGS_DB = os.path.join(PROJECT_ROOT, "data", "chat_logs.db")

st.set_page_config(page_title="Админ-панель ДДТ Союз", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")


@st.cache_resource
def get_logger():
    return ChatLogger(CHAT_LOGS_DB)


def fmt_time(iso_str):
    if not iso_str:
        return "—"
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%d.%m %H:%M:%S")
    except Exception:
        return iso_str[:19]


def main():
    st.title("📊 Админ-панель ДДТ «Союз»")

    if not os.path.exists(CHAT_LOGS_DB):
        st.warning("База логов ещё пуста. Запустите бота и сделайте несколько запросов.")
        st.code(f"Ожидаемый путь: {CHAT_LOGS_DB}")
        return

    logger = get_logger()

    # ─── Sidebar ───
    with st.sidebar:
        st.markdown("### Навигация")
        view = st.radio("Раздел", ["Обзор", "Сессии", "Дизлайки", "Лайки"])
        st.markdown("---")
        if st.button("🔄 Обновить данные"):
            st.cache_resource.clear()
            st.rerun()

    # ─── Обзор ───
    if view == "Обзор":
        st.subheader("Общая статистика")
        stats = logger.get_summary_stats()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Сессий", stats['total_sessions'])
        col2.metric("Сообщений", stats['total_messages'])
        col3.metric("👍 Лайков", stats['likes'])
        col4.metric("👎 Дизлайков", stats['dislikes'])

        col1, col2 = st.columns(2)
        col1.metric("От пользователей", stats['user_messages'])
        col2.metric("От бота", stats['assistant_messages'])

        st.metric("Средн. сообщений на сессию", stats['avg_messages_per_session'])

        if stats['likes'] + stats['dislikes'] > 0:
            satisfaction = stats['likes'] / (stats['likes'] + stats['dislikes']) * 100
            st.markdown(f"### Уровень удовлетворённости: **{satisfaction:.1f}%**")
            st.progress(satisfaction / 100)

    # ─── Сессии ───
    elif view == "Сессии":
        st.subheader("Последние сессии")
        sessions = logger.get_recent_sessions(limit=50)

        if not sessions:
            st.info("Сессий пока нет")
            return

        # Таблица
        for s in sessions:
            with st.expander(
                f"🕐 {fmt_time(s['started_at'])} · "
                f"💬 {s['message_count']} сообщ. · "
                f"👍{s['feedback_likes']} 👎{s['feedback_dislikes']} · "
                f"`{s['session_id'][:8]}`"
            ):
                msgs = logger.get_session_messages(s['session_id'])
                if not msgs:
                    st.text("(пусто)")
                    continue

                for m in msgs:
                    if m['role'] == 'user':
                        st.markdown(f"**👤 Пользователь** _{fmt_time(m['timestamp'])}_")
                        st.markdown(f"> {m['content']}")
                    else:
                        fb = ""
                        if m['feedback'] == 'like':
                            fb = " 👍"
                        elif m['feedback'] == 'dislike':
                            fb = " 👎"
                        st.markdown(f"**🤖 Бот**{fb} _{fmt_time(m['timestamp'])}_")

                        # Метаданные ответа
                        meta_parts = []
                        if m.get('response_time_seconds'):
                            meta_parts.append(f"{m['response_time_seconds']:.1f}с")
                        if m.get('model_used'):
                            meta_parts.append(m['model_used'])
                        if m.get('llm_calls'):
                            meta_parts.append(f"LLM×{m['llm_calls']}")
                        if m.get('keywords'):
                            try:
                                kws = json.loads(m['keywords'])
                                meta_parts.append(f"kw: {', '.join(kws[:5])}")
                            except Exception:
                                pass
                        if meta_parts:
                            st.caption(" | ".join(meta_parts))

                        st.markdown(m['content'])
                    st.markdown("---")

    # ─── Дизлайки ───
    elif view == "Дизлайки":
        st.subheader("Ответы с дизлайком (требуют внимания)")
        disliked = logger.get_disliked_messages(limit=100)

        if not disliked:
            st.success("Дизлайков пока нет!")
            return

        st.info(f"Найдено: {len(disliked)}")

        for d in disliked:
            with st.container():
                st.markdown(f"**🕐 {fmt_time(d.get('feedback_timestamp', d['timestamp']))}**")
                st.markdown(f"**👤 Вопрос:** {d.get('user_question', '(не найден)')}")

                meta_parts = []
                if d.get('response_time_seconds'):
                    meta_parts.append(f"{d['response_time_seconds']:.1f}с")
                if d.get('model_used'):
                    meta_parts.append(d['model_used'])
                if d.get('keywords'):
                    try:
                        kws = json.loads(d['keywords'])
                        meta_parts.append(f"kw: {', '.join(kws[:5])}")
                    except Exception:
                        pass
                if meta_parts:
                    st.caption(" | ".join(meta_parts))

                st.markdown(f"**🤖 Ответ:**")
                st.markdown(f"> {d['content']}")
                st.caption(f"Сессия: `{d['session_id'][:8]}`")
                st.markdown("---")

    # ─── Лайки ───
    elif view == "Лайки":
        st.subheader("Ответы с лайком (что работает хорошо)")
        liked = logger.get_liked_messages(limit=100)

        if not liked:
            st.info("Лайков пока нет")
            return

        st.success(f"Найдено: {len(liked)}")

        for d in liked:
            with st.container():
                st.markdown(f"**🕐 {fmt_time(d.get('feedback_timestamp', d['timestamp']))}**")
                st.markdown(f"**👤 Вопрос:** {d.get('user_question', '(не найден)')}")

                meta_parts = []
                if d.get('response_time_seconds'):
                    meta_parts.append(f"{d['response_time_seconds']:.1f}с")
                if d.get('model_used'):
                    meta_parts.append(d['model_used'])
                if d.get('keywords'):
                    try:
                        kws = json.loads(d['keywords'])
                        meta_parts.append(f"kw: {', '.join(kws[:5])}")
                    except Exception:
                        pass
                if meta_parts:
                    st.caption(" | ".join(meta_parts))

                st.markdown(f"**🤖 Ответ:**")
                st.markdown(f"> {d['content'][:500]}{'...' if len(d['content']) > 500 else ''}")
                st.caption(f"Сессия: `{d['session_id'][:8]}`")
                st.markdown("---")


if __name__ == "__main__":
    main()