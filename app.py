"""
app.py — Streamlit веб-интерфейс для RAG чат-бота ДДТ «Союз».
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging
logging.getLogger("streamlit.watcher").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime").setLevel(logging.ERROR)

import streamlit as st
import time, json, sqlite3

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config.prompts import MODEL_PRIORITY, GREETING_RESPONSE, EMBEDDING_MODEL
from services.api_key_manager.api_key_manager import SmartKeyManager, Logger, load_api_keys
from services.rag.rag_chatbot import RAGChatbot
from services.rag import search as search_module
from services.logging.chat_logger import ChatLogger

st.set_page_config(page_title="ДДТ «Союз» — Подбор программ", page_icon="🎨",
                   layout="centered", initial_sidebar_state="expanded")

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LOGO_PATH = os.path.join(DATA_DIR, "logo_streamlit.png")
DB_PATH = os.path.join(PROJECT_ROOT, "services", "db", "knowledge.db")
STATS_FILE = os.path.join(PROJECT_ROOT, "services", "api_key_manager", "key_stats.json")
CHAT_LOGS_DB = os.path.join(DATA_DIR, "chat_logs.db")
ADMIN_PASSWORD = "ddt2026"

QUICK_ANSWERS_PATH = os.path.join(PROJECT_ROOT, "services", "rag", "quick_answers.json")
QUICK_ANSWERS = {}
if os.path.exists(QUICK_ANSWERS_PATH):
    with open(QUICK_ANSWERS_PATH, 'r', encoding='utf-8') as f:
        QUICK_ANSWERS = json.load(f)

FONT_IMPORT = '@import url("https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;500;600;700&display=swap");'
FONT_FAMILY = '"Nunito", sans-serif'
CHAT_MAX_WIDTH = 820

CUSTOM_CSS = """<style>
%s
.stApp{background:linear-gradient(135deg,#EBF4FF 0%%,#F0F7FF 40%%,#E8F0FE 100%%);font-family:%s;color:#1a3a5c;}
footer{visibility:hidden;}.stDeployButton{display:none;}
.block-container{max-width:%dpx!important;padding-left:1rem!important;padding-right:1rem!important;}
.header-container{display:flex;align-items:center;gap:16px;padding:20px 0 16px 0;border-bottom:1px solid #c8ddf0;margin-bottom:20px;}
.header-logo{width:56px;height:56px;border-radius:14px;object-fit:cover;box-shadow:0 2px 8px rgba(0,0,0,0.08);}
.header-title{font-family:%s;font-size:22px;font-weight:600;color:#1a3a5c;line-height:1.2;}
.header-subtitle{font-family:%s;font-size:13px;color:#5a7a9a;font-weight:400;margin-top:2px;}
.stChatMessage{font-family:%s;font-size:15px;line-height:1.6;color:#1a3a5c;
  border-radius:16px!important;padding:12px 16px!important;margin-bottom:8px!important;
  word-wrap:break-word!important;overflow-wrap:break-word!important;max-width:100%%!important;overflow-x:hidden!important;}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]){
  background:linear-gradient(135deg,#3a6ea5 0%%,#2c5282 100%%)!important;color:white!important;
  border-radius:18px 18px 4px 18px!important;margin-left:48px!important;}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) p{color:white!important;}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]){
  background:rgba(255,255,255,0.92)!important;border:1px solid #c8ddf0!important;
  border-radius:18px 18px 18px 4px!important;margin-right:48px!important;
  box-shadow:0 1px 3px rgba(0,0,0,0.04);color:#1a3a5c!important;}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) p{color:#1a3a5c!important;}
.stChatInput{border-radius:24px!important;}
.stChatInput>div{border-radius:24px!important;border:2px solid #c8ddf0!important;background:rgba(255,255,255,0.95)!important;}
.stChatInput input{font-family:%s!important;font-size:15px!important;color:#1a3a5c!important;}
section[data-testid="stSidebar"]{background:#e8f0fe;font-family:%s;color:#1a3a5c;}
div[data-testid="column"] button[kind="secondary"]{font-size:14px!important;padding:2px 10px!important;
  min-height:28px!important;height:28px!important;border-radius:14px!important;
  background:rgba(255,255,255,0.7)!important;border:1px solid #c8ddf0!important;color:#5a7a9a!important;}
</style>""" % (FONT_IMPORT, FONT_FAMILY, CHAT_MAX_WIDTH,
               FONT_FAMILY, FONT_FAMILY, FONT_FAMILY, FONT_FAMILY, FONT_FAMILY)


# ═══════════════════════════════════════════════════════════════════════════════
# НАПРАВЛЕНИЯ
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_directions():
    dirs = []
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute('SELECT DISTINCT direction FROM programs WHERE direction IS NOT NULL')
        for r in cur.fetchall():
            d = r[0].strip()
            if d and 'модуль' not in d.lower():
                dirs.append(d)
        conn.close()
    except Exception:
        pass
    return sorted(set(dirs))


SCHOOL_DIRECTIONS = None


def get_school_response():
    global SCHOOL_DIRECTIONS
    if SCHOOL_DIRECTIONS is None:
        SCHOOL_DIRECTIONS = load_directions()
    return ("У нас много интересного для школьников! Уточните, вы бы хотели увидеть "
            "**список всех занятий** или по конкретному направлению?\n\n"
            "Доступные направления: %s.\n\n"
            "Просто напишите интересующее направление или \"все\"." % ", ".join(SCHOOL_DIRECTIONS))


SIGNUP_RESPONSE = (
    "Записаться на занятия в ДДТ «Союз» можно двумя способами:\n\n"
    "**1. Через «Навигатор дополнительного образования»** — "
    "[dopobr.petersburgedu.ru](https://dopobr.petersburgedu.ru/organizations/9474/) "
    "(для бюджетных программ)\n\n"
    "**2. Через форму записи** — для платных и некоторых бюджетных программ\n\n"
    "Чтобы я помог с записью, подскажите:\n"
    "- **Название программы** (например, «Забава», «Белый ферзь», «Рондо»)\n"
    "- или **Фамилию педагога** (например, Парёха, Дульян, Иванов)\n\n"
    "Я покажу информацию о программе и ссылку для записи."
)


# ═══════════════════════════════════════════════════════════════════════════════
# ИНИЦИАЛИЗАЦИЯ
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_ml_models():
    emb_model = None
    try:
        print("[app] Загрузка %s..." % EMBEDDING_MODEL)
        t = time.time()
        from sentence_transformers import SentenceTransformer
        emb_model = SentenceTransformer(EMBEDDING_MODEL)
        print("[app] %s загружен за %.1fs" % (EMBEDDING_MODEL, time.time() - t))
    except Exception as e:
        print("[app] эмбеддинг-модель недоступна: %s" % e)
    return emb_model, None


@st.cache_resource
def init_bot():
    emb_model, _ = load_ml_models()
    search_module.init_models(embedding_model=emb_model, reranker_model=None)

    if not os.path.exists(DB_PATH):
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "services", "db"))
        from build_db import create_database
        create_database(os.path.join(DATA_DIR, "programs.xlsx"),
                        os.path.join(DATA_DIR, "FAQ.docx"), DB_PATH)

    keys = load_api_keys(PROJECT_ROOT)
    if not keys:
        st.error("API-ключи не найдены.")
        st.stop()

    logger = Logger("dev")
    km = SmartKeyManager(keys, logger, cooldown_seconds=60, stats_file=STATS_FILE)
    chat_logger = ChatLogger(CHAT_LOGS_DB)
    print("[app] Инициализирован: %d ключей" % km.total_keys)
    return DB_PATH, km, logger, chat_logger


def get_bot():
    if "bot" not in st.session_state:
        db, km, log, chat_logger = init_bot()
        st.session_state.bot = RAGChatbot(db, km, log, MODEL_PRIORITY)
        st.session_state.chat_logger = chat_logger
        st.session_state.session_id = chat_logger.create_session()
    return st.session_state.bot


# ═══════════════════════════════════════════════════════════════════════════════
# ОТРИСОВКА
# ═══════════════════════════════════════════════════════════════════════════════

def render_header():
    if os.path.exists(LOGO_PATH):
        import base64
        with open(LOGO_PATH, "rb") as f:
            ld = base64.b64encode(f.read()).decode()
        logo = '<img src="data:image/png;base64,%s" class="header-logo">' % ld
    else:
        logo = ('<div class="header-logo" style="background:#3a6ea5;display:flex;'
                'align-items:center;justify-content:center;color:white;'
                'font-size:24px;font-weight:700;">С</div>')
    st.markdown(
        '<div class="header-container">%s<div>'
        '<div class="header-title">Дом детского творчества «Союз»</div>'
        '<div class="header-subtitle">Помощник по выбору образовательных программ</div>'
        '</div></div>' % logo, unsafe_allow_html=True)


def render_feedback_buttons(message_idx, log_message_id):
    chat_logger = st.session_state.chat_logger
    current = st.session_state.messages[message_idx].get("_feedback")
    col1, col2, col3 = st.columns([1, 1, 10])
    like_label = "👍✓" if current == "like" else "👍"
    dislike_label = "👎✓" if current == "dislike" else "👎"
    if col1.button(like_label, key="like_%d" % message_idx):
        new_val = None if current == "like" else "like"
        chat_logger.set_feedback(log_message_id, new_val)
        st.session_state.messages[message_idx]["_feedback"] = new_val
        st.rerun()
    if col2.button(dislike_label, key="dislike_%d" % message_idx):
        new_val = None if current == "dislike" else "dislike"
        chat_logger.set_feedback(log_message_id, new_val)
        st.session_state.messages[message_idx]["_feedback"] = new_val
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# ФОРМА ОБРАТНОЙ СВЯЗИ
# ═══════════════════════════════════════════════════════════════════════════════

def render_survey_form():
    """Форма обратной связи."""
    chat_logger = st.session_state.chat_logger
    session_id = st.session_state.session_id

    if st.session_state.get("_survey_submitted"):
        st.success("Спасибо за вашу обратную связь! 🙏")
        return

    with st.expander("📝 Обратная связь", expanded=st.session_state.get("_survey_open", False)):
        st.markdown("Пожалуйста, оцените работу нашего помощника:")

        q1 = st.slider("Оцените удовлетворённость ответами помощника",
                        1, 5, 3, key="survey_q1",
                        help="5 — отлично, 1 — плохо")

        q3 = st.slider("Насколько вероятно, что вы посоветуете помощника другим родителям?",
                        1, 5, 3, key="survey_q3",
                        help="5 — обязательно посоветую, 1 — точно нет")

        q5 = st.slider("Насколько релевантными были ответы помощника?",
                        1, 5, 3, key="survey_q5",
                        help="5 — точно то, что нужно, 1 — совсем не то")

        comment = st.text_area("Ваш комментарий (необязательно)",
                               key="survey_comment", height=80,
                               placeholder="Напишите ваши пожелания или замечания...")

        if st.button("Отправить отзыв", key="survey_submit", use_container_width=True):
            chat_logger.save_survey(session_id, q1, None, q3, None, q5, comment=comment or None)
            st.session_state._survey_submitted = True
            print("[app] Опрос: q1=%d q3=%d q5=%d comment=%s session=%s" %
                  (q1, q3, q5, repr(comment[:50]) if comment else 'None', session_id[:8]))
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# АДМИН-ПАНЕЛЬ
# ═══════════════════════════════════════════════════════════════════════════════

def _render_message_full(m):
    """Одно сообщение с полной debug-информацией."""
    if m['role'] == 'user':
        st.markdown("**👤 Пользователь:**")
        st.markdown(m['content'])
    else:
        fb = ""
        if m.get('feedback') == 'like':
            fb = " 👍"
        elif m.get('feedback') == 'dislike':
            fb = " 👎"
        st.markdown("**🤖 Бот%s:**" % fb)
        st.markdown(m['content'])

        # Debug-информация
        debug_parts = []
        if m.get('response_time_seconds') is not None:
            debug_parts.append("⏱ %.1fs" % m['response_time_seconds'])
        if m.get('model_used'):
            debug_parts.append("🧠 %s" % m['model_used'])
        if m.get('llm_calls') is not None:
            debug_parts.append("LLM×%d" % m['llm_calls'])
        if m.get('keywords'):
            try:
                kws = json.loads(m['keywords'])
                if kws:
                    debug_parts.append("🔑 %s" % ', '.join(str(k) for k in kws))
            except Exception:
                pass
        if m.get('age_collected'):
            debug_parts.append("👶 %d лет" % m['age_collected'])
        if debug_parts:
            st.caption(" | ".join(debug_parts))

    st.markdown("---")


def _render_session_full(chat_logger, session_id):
    """Полная переписка сессии с debug."""
    msgs = chat_logger.get_session_messages(session_id)
    if not msgs:
        st.text("(пусто)")
        return
    for m in msgs:
        _render_message_full(m)


def render_admin_panel():
    st.markdown("## 📊 Панель администратора")

    if not os.path.exists(CHAT_LOGS_DB):
        st.info("Логи пока пусты.")
        return

    chat_logger = ChatLogger(CHAT_LOGS_DB)
    stats = chat_logger.get_summary_stats()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Сессий", stats['total_sessions'])
    c2.metric("Сообщений", stats['total_messages'])
    c3.metric("👍", stats['likes'])
    c4.metric("👎", stats['dislikes'])
    c5.metric("📝 Отзывов", stats.get('surveys', 0))

    if stats['likes'] + stats['dislikes'] > 0:
        sat = stats['likes'] / (stats['likes'] + stats['dislikes']) * 100
        st.markdown("**Удовлетворённость (лайки): %.0f%%**" % sat)
        st.progress(sat / 100)

    tab1, tab2, tab3, tab4 = st.tabs(["Последние сессии", "👎 Дизлайки", "👍 Лайки", "📝 Отзывы"])

    with tab1:
        sessions = chat_logger.get_recent_sessions(limit=30)
        for s in sessions:
            ts = s['started_at'][:16] if s.get('started_at') else '?'
            with st.expander("🕐 %s · 💬%d · 👍%d 👎%d · `%s`" %
                             (ts, s['message_count'],
                              s['feedback_likes'], s['feedback_dislikes'],
                              s['session_id'][:8])):
                _render_session_full(chat_logger, s['session_id'])

    with tab2:
        disliked = chat_logger.get_disliked_messages(limit=50)
        if not disliked:
            st.success("Дизлайков нет!")
        for d in disliked:
            with st.expander("👎 %s · `%s`" %
                             (d.get('feedback_timestamp', '')[:16], d['session_id'][:8])):
                _render_session_full(chat_logger, d['session_id'])

    with tab3:
        liked = chat_logger.get_liked_messages(limit=50)
        if not liked:
            st.info("Лайков пока нет")
        for d in liked:
            with st.expander("👍 %s · `%s`" %
                             (d.get('feedback_timestamp', '')[:16], d['session_id'][:8])):
                _render_session_full(chat_logger, d['session_id'])

    with tab4:
        # Средние оценки
        avgs = chat_logger.get_survey_averages()
        if avgs and avgs['cnt'] > 0:
            st.markdown("### Средние оценки (%d отзывов)" % avgs['cnt'])
            col1, col2 = st.columns(2)
            col1.metric("Удовлетворённость", "%.1f / 5" % (avgs['avg_q1'] or 0))
            col2.metric("Рекомендация", "%.1f / 5" % (avgs['avg_q3'] or 0))
            col1.metric("Скорость", "%.1f / 5" % (avgs['avg_q4'] or 0))
            col2.metric("Релевантность", "%.1f / 5" % (avgs['avg_q5'] or 0))
            if avgs['q2_yes'] + avgs['q2_no'] > 0:
                pct = avgs['q2_yes'] / (avgs['q2_yes'] + avgs['q2_no']) * 100
                st.markdown("**Удобнее с помощником:** %.0f%% Да, %.0f%% Нет" %
                            (pct, 100 - pct))
            st.markdown("---")

        surveys = chat_logger.get_surveys(limit=50)
        if not surveys:
            st.info("Отзывов пока нет")
        for s in surveys:
            ts = s['timestamp'][:16] if s.get('timestamp') else '?'
            st.markdown("**%s** · сессия `%s` · сообщений: %s" %
                        (ts, s['session_id'][:8], s.get('message_count', '?')))
            st.markdown("Удовлетворённость: **%s**/5 · "
                        "Рекомендация: **%s**/5 · "
                        "Релевантность: **%s**/5" %
                        (s['q1_satisfaction'], s['q3_recommend'], s['q5_relevance']))
            if s.get('comment'):
                st.markdown("💬 *%s*" % s['comment'])
            st.markdown("---")

    # Статистика ключей
    if st.button("📈 Статистика API-ключей"):
        bot = get_bot()
        st.code(bot.key_manager.get_stats_report())


# ═══════════════════════════════════════════════════════════════════════════════
# ГЛАВНАЯ
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "admin_mode" not in st.session_state:
        st.session_state.admin_mode = False
    if "admin_auth" not in st.session_state:
        st.session_state.admin_auth = False

    # ─── Сайдбар ───
    with st.sidebar:
        st.markdown("### Быстрые вопросы")
        qs = list(QUICK_ANSWERS.keys()) if QUICK_ANSWERS else []
        for i, q in enumerate(qs):
            if st.button(q, key="quick_%d" % i, use_container_width=True):
                st.session_state._quick_q = q
                st.rerun()

        st.markdown("---")
        st.markdown("### Настройки")

        bot = get_bot()
        chat_logger = st.session_state.chat_logger
        session_id = st.session_state.session_id

        if st.button("🗑 Очистить историю", use_container_width=True):
            st.session_state.messages = []
            bot.reset()
            st.session_state.session_id = chat_logger.create_session()
            st.session_state._survey_submitted = False
            st.rerun()

        st.markdown("---")
        if not st.session_state.admin_auth:
            with st.expander("🔐 Вход для администратора"):
                pwd = st.text_input("Пароль", type="password", key="admin_pwd_input")
                if st.button("Войти", key="admin_login_btn"):
                    if pwd == ADMIN_PASSWORD:
                        st.session_state.admin_auth = True
                        st.session_state.admin_mode = True
                        st.rerun()
                    else:
                        st.error("Неверный пароль")
        else:
            if st.checkbox("📊 Админ-панель", value=st.session_state.admin_mode):
                st.session_state.admin_mode = True
            else:
                st.session_state.admin_mode = False
            if st.button("🚪 Выйти", use_container_width=True):
                st.session_state.admin_auth = False
                st.session_state.admin_mode = False
                st.rerun()

        st.markdown("---")
        st.markdown("### Контакты")
        st.markdown("📞 8 995 834 09 94")
        st.markdown("📍 пр. Раевского, 5/2")
        st.markdown("📍 пр. Тореза, 35/1")

    # ─── Админ-режим ───
    if st.session_state.admin_mode:
        render_admin_panel()
        return

    # ─── Обычный чат ───
    render_header()

    if not st.session_state.messages:
        st.session_state.messages.append({
            "role": "assistant", "content": GREETING_RESPONSE, "_log_id": None,
        })

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("_log_id"):
                render_feedback_buttons(i, msg["_log_id"])

    # Форма обратной связи
    render_survey_form()

    quick_q = st.session_state.get("_quick_q")
    if quick_q:
        st.session_state._quick_q = None
    user_input = st.chat_input("Задайте вопрос о кружках и программах...")
    prompt = user_input or quick_q

    if prompt:
        chat_logger.log_message(session_id=session_id, role="user", content=prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        quick_val = QUICK_ANSWERS.get(prompt)

        with st.chat_message("assistant"):
            if quick_val == "__ask_school__":
                response = get_school_response()
                elapsed = 0.0
                calls_made = 0
                model_used = "template"
                if bot.collected_age is None:
                    bot.collected_age = 7
                    bot._exclude_preschool = True
                st.markdown(response)

            elif quick_val == "__ask_signup__":
                response = SIGNUP_RESPONSE
                elapsed = 0.0
                calls_made = 0
                model_used = "template"
                st.markdown(response)

            elif quick_val and not quick_val.startswith("__"):
                response = quick_val
                elapsed = 0.0
                calls_made = 0
                model_used = "quick_answers"
                st.markdown(response)

            else:
                with st.spinner("Ищу подходящий ответ..."):
                    try:
                        start = time.time()
                        calls_before = bot.llm_calls
                        response = bot.process_message(prompt)
                        elapsed = time.time() - start
                        calls_made = bot.llm_calls - calls_before
                        model_used = bot.current_model if calls_made > 0 else "без LLM"
                        print("[app] %.1fs | %s | LLM x%d | kw: %s" %
                              (elapsed, model_used, calls_made, bot.collected_keywords))
                    except Exception as e:
                        response = ("Извините, произошла ошибка. "
                                    "Попробуйте ещё раз или позвоните: 8 995 834 09 94.")
                        model_used = "error"
                        elapsed = 0.0
                        calls_made = 0
                        print("[app] ОШИБКА: %s" % str(e))
                st.markdown(response)

        log_id = chat_logger.log_message(
            session_id=session_id, role="assistant", content=response,
            llm_calls=calls_made, model_used=model_used,
            response_time_seconds=elapsed,
            keywords=list(bot.collected_keywords) if bot.collected_keywords else None,
            age_collected=bot.collected_age,
            served_directions=getattr(bot, 'last_served_directions', None) or None)

        st.session_state.messages.append({
            "role": "assistant", "content": response, "_log_id": log_id,
        })
        st.rerun()


if __name__ == "__main__":
    main()