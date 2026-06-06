"""
server.py — Flask-сервер для RAG чат-бота ДДТ «Союз».
Заменяет Streamlit. Обслуживает Vue.js фронтенд + API.
"""
import os, sys, warnings, json, time, re
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, request, jsonify, send_from_directory
from config.prompts import MODEL_PRIORITY, GREETING_RESPONSE, EMBEDDING_MODEL
from services.api_key_manager.api_key_manager import SmartKeyManager, Logger, load_api_keys
from services.rag.rag_chatbot import RAGChatbot
from services.rag import search as search_module
from services.logging.chat_logger import ChatLogger

app = Flask(__name__, static_folder='static')

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(PROJECT_ROOT, "services", "db", "knowledge.db")
STATS_FILE = os.path.join(PROJECT_ROOT, "services", "api_key_manager", "key_stats.json")
CHAT_LOGS_DB = os.path.join(DATA_DIR, "chat_logs.db")
ADMIN_PASSWORD = "ddt2026"

QUICK_ANSWERS_PATH = os.path.join(PROJECT_ROOT, "services", "rag", "quick_answers.json")
QUICK_ANSWERS = {}
if os.path.exists(QUICK_ANSWERS_PATH):
    with open(QUICK_ANSWERS_PATH, 'r', encoding='utf-8') as f:
        QUICK_ANSWERS = json.load(f)

# ═══════════════════════════════════════════════════════════════════════════════
# ИНИЦИАЛИЗАЦИЯ
# ═══════════════════════════════════════════════════════════════════════════════

print("[server] Загрузка моделей...")
t = time.time()

# Пересборка БД если удалена
if not os.path.exists(DB_PATH):
    print("[server] knowledge.db не найдена — пересобираю...")
    try:
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "services", "db"))
        from build_db import create_database
        xlsx = os.path.join(DATA_DIR, "programs.xlsx")
        faq  = os.path.join(DATA_DIR, "FAQ.docx")
        create_database(xlsx, faq, DB_PATH)
        print("[server] knowledge.db создана")
        # Пересборка векторов
        try:
            from build_vectors import main as build_vectors_main
            build_vectors_main()
            print("[server] Векторы пересобраны")
        except Exception as ve:
            print("[server] Векторы не пересобраны: %s" % ve)
    except Exception as e:
        print("[server] ОШИБКА пересборки БД: %s" % e)
try:
    from sentence_transformers import SentenceTransformer
    emb_model = SentenceTransformer(EMBEDDING_MODEL)
    print("[server] %s загружен за %.1fs" % (EMBEDDING_MODEL, time.time() - t))
except Exception as e:
    emb_model = None
    print("[server] эмбеддинг-модель недоступна: %s" % e)

search_module.init_models(embedding_model=emb_model, reranker_model=None)

keys = load_api_keys(PROJECT_ROOT)
logger = Logger("dev")
km = SmartKeyManager(keys, logger, cooldown_seconds=60, stats_file=STATS_FILE)
chat_logger = ChatLogger(CHAT_LOGS_DB)

print("[server] Инициализирован: %d ключей, модели: %s" % (km.total_keys, MODEL_PRIORITY))

# Пул ботов по session_id (ограниченный)
_bots = {}
_MAX_BOTS = 50


def get_bot(session_id):
    if session_id not in _bots:
        if len(_bots) > _MAX_BOTS:
            oldest = list(_bots.keys())[0]
            del _bots[oldest]
        _bots[session_id] = RAGChatbot(DB_PATH, km, logger, MODEL_PRIORITY)
    return _bots[session_id]


# ═══════════════════════════════════════════════════════════════════════════════
# СТАТИКА
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/admin')
def admin():
    return send_from_directory('static', 'admin.html')


@app.route('/static/<path:path>')
def static_files(path):
    return send_from_directory('static', path)


# ═══════════════════════════════════════════════════════════════════════════════
# API
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.json or {}
    message = data.get('message', '').strip()
    session_id = data.get('session_id', 'default')

    if not message:
        return jsonify({'response': '', 'error': 'empty message'})

    bot = get_bot(session_id)

    # Автоматически создаём сессию если нет
    sess = chat_logger.get_session_stats(session_id)
    if not sess:
        # Создаём сессию с этим ID
        from datetime import datetime
        with chat_logger._conn() as c:
            c.execute("""INSERT OR IGNORE INTO sessions
                (session_id, started_at, last_active) VALUES (?, ?, ?)""",
                (session_id, datetime.now().isoformat(), datetime.now().isoformat()))
            c.commit()

    # Спец.кнопки из quick_answers.json
    quick_val = QUICK_ANSWERS.get(message)
    if quick_val:
        # Особая логика для "Занятия для школьников" — устанавливаем возраст
        if message == "Занятия для школьников":
            if bot.collected_age is None:
                bot.collected_age = 7
                bot._exclude_preschool = True

        # На всякий случай: сбрасываем атрибут last_served_directions, чтобы
        # никакой остаток от предыдущего хода не "перетёк" в логику отображения.
        # В саму БД ответ кнопки идёт с served_directions=None — кнопочные
        # ответы не считаются за выдачу программ.
        bot.last_served_directions = []

        chat_logger.log_message(session_id=session_id, role="user", content=message)
        log_id = chat_logger.log_message(
            session_id=session_id, role="assistant", content=quick_val,
            llm_calls=0, model_used="quick_answers", response_time_seconds=0,
            served_directions=None)
        return jsonify({
            'response': quick_val,
            'log_id': log_id,
            'meta': {'category': 'quick', 'llm_calls': 0}
        })

    # Логируем
    chat_logger.log_message(session_id=session_id, role="user", content=message)

    try:
        t_start = time.time()
        calls_before = bot.llm_calls
        response = bot.process_message(message)
        elapsed = time.time() - t_start
        calls_made = bot.llm_calls - calls_before
        model_used = bot.current_model if calls_made > 0 else "без LLM"

        log_id = chat_logger.log_message(
            session_id=session_id, role="assistant", content=response,
            llm_calls=calls_made, model_used=model_used,
            response_time_seconds=elapsed,
            keywords=list(bot.collected_keywords) if bot.collected_keywords else None,
            age_collected=bot.collected_age,
            served_directions=getattr(bot, 'last_served_directions', None) or None)

        print("[server] %.1fs | %s | LLM x%d | kw: %s" %
              (elapsed, model_used, calls_made, bot.collected_keywords))

        return jsonify({
            'response': response,
            'log_id': log_id,
            'meta': {
                'elapsed': round(elapsed, 1),
                'model': model_used,
                'llm_calls': calls_made,
                'keywords': list(bot.collected_keywords),
                'age': bot.collected_age,
            }
        })
    except Exception as e:
        print("[server] ОШИБКА: %s" % str(e))
        return jsonify({
            'response': 'Извините, произошла ошибка. Позвоните: 8 995 834 09 94.',
            'meta': {'error': str(e)[:200]}
        })


@app.route('/api/reset', methods=['POST'])
def api_reset():
    data = request.json or {}
    session_id = data.get('session_id', 'default')
    if session_id in _bots:
        _bots[session_id].reset()
    new_sid = chat_logger.create_session()
    return jsonify({'session_id': new_sid})


@app.route('/api/feedback', methods=['POST'])
def api_feedback():
    data = request.json or {}
    log_id = data.get('log_id')
    feedback = data.get('feedback')
    if log_id:
        chat_logger.set_feedback(log_id, feedback)
    return jsonify({'ok': True})


@app.route('/api/survey', methods=['POST'])
def api_survey():
    data = request.json or {}
    chat_logger.save_survey(
        data.get('session_id', ''),
        data.get('q1'), None, data.get('q3'), None, data.get('q5'),
        comment=data.get('comment'))
    return jsonify({'ok': True})


# ═══════════════════════════════════════════════════════════════════════════════
# АДМИН API
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/admin/login', methods=['POST'])
def admin_login():
    data = request.json or {}
    if data.get('password') == ADMIN_PASSWORD:
        return jsonify({'ok': True})
    return jsonify({'ok': False}), 403


@app.route('/api/admin/stats', methods=['GET'])
def admin_stats():
    stats = chat_logger.get_summary_stats()
    avgs = chat_logger.get_survey_averages()
    return jsonify({'stats': stats, 'survey_averages': avgs})


@app.route('/api/admin/sessions', methods=['GET'])
def admin_sessions():
    """
    Сессии с фильтрами и пагинацией.
    Query-параметры (все опциональны):
      page (default 1), page_size (default 15, max 100),
      date_from / date_to (YYYY-MM-DD),
      duration_min / duration_max (секунды),
      user_msg_min / user_msg_max,
      likes_min / likes_max, dislikes_min / dislikes_max,
      directions (CSV-список названий направлений).
    """
    args = request.args

    def _int(name, default=None):
        v = args.get(name)
        if v is None or v == '':
            return default
        try:
            return int(v)
        except ValueError:
            return default

    def _str(name):
        v = args.get(name)
        return v.strip() if v and v.strip() else None

    dirs_csv = _str('directions')
    directions = [d.strip() for d in dirs_csv.split(',')] if dirs_csv else None

    page_size = _int('page_size', 15) or 15
    page_size = max(1, min(page_size, 100))

    result = chat_logger.get_sessions_filtered(
        page=_int('page', 1) or 1,
        page_size=page_size,
        date_from=_str('date_from'),
        date_to=_str('date_to'),
        duration_min=_int('duration_min'),
        duration_max=_int('duration_max'),
        user_msg_min=_int('user_msg_min'),
        user_msg_max=_int('user_msg_max'),
        likes_min=_int('likes_min'),
        likes_max=_int('likes_max'),
        dislikes_min=_int('dislikes_min'),
        dislikes_max=_int('dislikes_max'),
        directions=directions,
    )
    return jsonify(result)


@app.route('/api/admin/directions', methods=['GET'])
def admin_directions():
    """Сводка популярности направлений + список доступных названий для фильтра.
    Источник — поле messages.served_directions, заполняемое ботом для тех ответов,
    в которых были реально выданы программы. FAQ и шаблоны не учитываются."""
    args = request.args
    data = chat_logger.get_directions_summary(
        date_from=(args.get('date_from') or None),
        date_to=(args.get('date_to') or None),
    )
    return jsonify(data)


@app.route('/api/admin/disliked', methods=['GET'])
def admin_disliked():
    return jsonify(chat_logger.get_disliked_messages(limit=50))


@app.route('/api/admin/liked', methods=['GET'])
def admin_liked():
    return jsonify(chat_logger.get_liked_messages(limit=50))


@app.route('/api/admin/surveys', methods=['GET'])
def admin_surveys():
    return jsonify(chat_logger.get_surveys(limit=50))


@app.route('/api/admin/key_stats', methods=['GET'])
def admin_key_stats():
    return jsonify({'report': km.get_stats_report()})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8501, debug=False)