# ИИ-ассистент ДДТ «Союз»

Чат-бот для подбора образовательных программ Дома детского творчества «Союз» (Санкт-Петербург). Помогает родителям найти подходящую программу по возрасту, интересам, расписанию и площадке.

**Сайт:** [unionddt.ru](https://unionddt.ru)  
**Бот:** [bot.unionddt.ru](https://bot.unionddt.ru)  
**Телефон:** 8 995 834 09 94 (ПН-ПТ 11:00–19:00)  
**Площадки:** пр. Раевского 5/2 и пр. Тореза 35/1

---

## Содержание

- [Архитектура](#архитектура)
- [Структура проекта](#структура-проекта)
- [Быстрый старт](#быстрый-старт)
- [Сборка базы данных](#сборка-базы-данных)
- [Формат данных](#формат-данных)
- [Сценарии обработки запросов](#сценарии-обработки-запросов)
- [Конфигурация](#конфигурация)
- [Деплой на сервер](#деплой-на-сервер)
- [Обновление данных](#обновление-данных)
- [Безопасность](#безопасность)
- [Команды управления](#команды-управления)

---

## Архитектура

```
Пользователь
    │
    ▼
unionddt.ru  (WordPress + iframe-виджет)
    │  /?embed=1
    ▼
bot.unionddt.ru  (nginx + Let's Encrypt)
    │
    ▼
server.py  (Flask API, порт 8501)
    │
    ├──▶  RAGChatbot  (services/rag/rag_chatbot.py)
    │         │
    │         ├── Preprocessor (services/preprocessor/preprocessor.py)
    │         │       Извлечение возраста, направлений, ФИО, локации
    │         │
    │         ├── Search (services/rag/search.py)
    │         │       Гибридный поиск: FTS5 + sqlite-vec + reranker
    │         │
    │         └── SmartKeyManager (services/api_key_manager/api_key_manager.py)
    │                 Каскад Gemini-моделей, ротация ключей
    │
    ├──▶  ChatLogger (services/logging/chat_logger.py)
    │         SQLite: sessions, messages, surveys, feedback
    │
    └──▶  static/
              index.html   — виджет чата (Vue 3)
              admin.html   — панель администратора
```

### Стек технологий

| Компонент | Технология |
|-----------|-----------|
| API-сервер | Flask (Python 3.10+) |
| Поиск по тексту | SQLite FTS5 |
| Векторный поиск | sqlite-vec + intfloat/multilingual-e5-base (dim 768) |
| LLM | Google Gemini (каскад моделей) |
| Фронтенд виджета | Vue 3 (CDN, без сборщика) |
| Реверс-прокси | Nginx + Let's Encrypt |
| Автозапуск | systemd |
| БД логов | SQLite (chat_logs.db) |

---

## Структура проекта

```
ai_assistant_education/
│
├── server.py                        # Flask API (основная точка входа)
├── app.py                           # Streamlit-интерфейс (для разработки)
├── requirements.txt
│
├── config/
│   └── prompts.py                   # Промпты LLM, список моделей, приветствие
│
├── data/
│   ├── programs.xlsx                # Каталог программ (редактирует клиент)
│   ├── FAQ.docx                     # Вопросы и ответы (редактирует клиент)
│   └── chat_logs.db                 # Логи переписок (НЕ в git)
│
├── services/
│   ├── api_key_manager/
│   │   ├── api_key_manager.py       # SmartKeyManager + каскад моделей
│   │   └── api_accounts.xlsx        # Ключи Gemini API (НЕ в git)
│   │
│   ├── db/
│   │   ├── build_db.py              # Сборка knowledge.db из xlsx и docx
│   │   ├── build_vectors.py         # Генерация эмбеддингов (e5-base)
│   │   ├── validate_programs.py     # Валидатор programs.xlsx
│   │   ├── synonyms.json            # Словарь синонимов и тегов
│   │   └── knowledge.db             # Готовая база знаний (НЕ в git)
│   │
│   ├── logging/
│   │   └── chat_logger.py           # Логирование сессий, фидбека, опросов
│   │
│   ├── preprocessor/
│   │   └── preprocessor.py          # Извлечение возраста, направлений, ФИО
│   │
│   └── rag/
│       ├── rag_chatbot.py           # Основная логика бота (RAGChatbot)
│       ├── search.py                # Гибридный поиск по knowledge.db
│       └── quick_answers.json       # Ответы на кнопки быстрых вопросов
│
├── static/
│   ├── index.html                   # Виджет чата (Vue 3)
│   └── admin.html                   # Панель администратора
│
├── validation/                      # Оценочный датасет и скрипты
└── scripts/
    └── validate_programs.py         # Валидатор (для запуска из корня)
```

---

## Быстрый старт

### Требования

- Python 3.10+
- Ubuntu 22.04 / 24.04 LTS
- 4 ГБ RAM (минимум 2 ГБ)
- 10 ГБ диска
- Ключи Google Gemini API (`api_accounts.xlsx`)

### Установка

```bash
# 1. Клонировать репозиторий
git clone git@github.com:nonodoubt/ai_assistant_education.git bot
cd bot

# 2. Виртуальное окружение
python3 -m venv .venv
source .venv/bin/activate

# 3. Зависимости
pip install -r requirements.txt
pip install sentence-transformers

# 4. Положить секретные файлы (не в git)
#    services/api_key_manager/api_accounts.xlsx

# 5. Собрать БД
python services/db/build_db.py
python services/db/build_vectors.py

# 6. Запустить
python server.py
# → http://localhost:8501
```

---

## Сборка базы данных

### Полная пересборка

```bash
source .venv/bin/activate
python services/db/build_db.py      # валидация + сборка (2–3 сек)
python services/db/build_vectors.py # эмбеддинги (~2–5 мин, первый раз дольше)
sudo systemctl restart ddt-bot
```

### Только валидация

```bash
python services/db/validate_programs.py
python services/db/validate_programs.py /path/to/programs.xlsx  # произвольный файл
```

### Уровни сообщений валидатора

| Уровень | Значение |
|---------|----------|
| ❌ ОШИБКА | Сборка отменена — исправьте файл |
| ⚠️ ВНИМАНИЕ | Сборка продолжится, но качество поиска снизится |
| ℹ️ ИНФО | Рекомендация по улучшению данных |

---

## Формат данных

### programs.xlsx

Обязательные колонки (названия точь в точь, порядок не важен):

| Колонка | Пример |
|---------|--------|
| Направление | `Хореография` |
| Название коллектива | `Хореографический ансамбль «Забава»` |
| Возраст | `7-12 лет` или `5 лет` |
| Платное/Бюджет | `Бюджет` или `500 рублей / занятие` |
| Расписание | `ПН, СР 16:00–17:30` |
| Площадка | `пр. Раевского 5/2` |
| Что нужно для занятий | `Спортивная форма, сменная обувь` |
| Чему ребенок научится/ что изучит? | `Ритмика, координация, пластика` |
| Педагог | `Иванова Анна Сергеевна` |
| Информация о педагоге | `Стаж 15 лет, первая категория` |
| Ссылка на запись (навигатор) | `https://dopobr.petersburgedu.ru/...` |
| теги | `танцы, активность, осанка, ритм` |

**Допустимые значения:**
- **Площадка:** `пр. Раевского 5/2` или `пр Тореза 35/1`
- **Платное/Бюджет:** `Бюджет` или `500 рублей / занятие`
- **Возраст:** формат `N-M лет` или `N лет` (без пояснений в скобках)
- **Теги:** через запятую, строчными буквами

### FAQ.docx

Чередующиеся строки: **вопрос** (содержит `?`) — **ответ**. Таблицы поддерживаются.

```
Сколько стоит одно занятие?
Одно платное занятие стоит 500 рублей...

Есть ли пробные занятия?
Записаться на пробное занятие можно по телефону: 8 995 834 09 94.
```

---

## Сценарии обработки запросов

```
A   Мгновенные ответы    — приветствие, цифры возраста, смена темы
A2  Подтверждение        — «да»/«подробнее» после предложения альтернатив
B   Поиск по имени       — ФИО педагога, точное название программы
C   FAQ (гибридный)      — keyword-якоря + vector score ≥ 0.78
D   Поиск программ       — FTS5 + векторный поиск + фильтры возраст/локация/цена
E   Fallback             — уточняющий вопрос или общий ответ
```

### Шаг D — поиск программ

1. **Preprocessor** — возраст, направление, локация, день недели
2. **Keyword search** — FTS5 по тегам и тексту
3. **Vector search** — sqlite-vec, косинусное сходство
4. **Фильтры** — age_min ≤ age ≤ age_max, площадка, стоимость
5. **LLM** — формулировка ответа через Gemini

### Каскад моделей Gemini

```
gemini-3.5-flash → gemini-3.1-flash-lite → gemini-2.5-flash-lite → gemini-2.0-flash-lite
```

Таймаут на запрос — 12 сек. При транспортной ошибке — немедленный переход к следующей модели (не следующему ключу той же модели).

---

## Конфигурация

### config/prompts.py

```python
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"  # dim 768
EMBEDDING_DIM = 768
MODEL_PRIORITY = ["gemini-3.5-flash", "gemini-3.1-flash-lite", ...]
```

> ⚠️ При смене `EMBEDDING_MODEL` — обязательно `python services/db/build_vectors.py`

### Переменные окружения

Задаются в `/etc/systemd/system/ddt-bot.service`:

```ini
Environment="ADMIN_PASSWORD=пароль-формы-входа"
Environment="SECRET_KEY=случайная-строка-32-байта"
```

Генерация SECRET_KEY:
```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

---

## Деплой на сервер

### Требования к серверу

- Ubuntu 22.04 / 24.04 LTS, 4 ГБ RAM, 30 ГБ NVMe
- Hostland VDS Standart Middle или аналог

### Первичная установка

```bash
apt update && apt upgrade -y
apt install -y python3 python3-venv python3-pip nginx git certbot python3-certbot-nginx ufw

ufw allow 22/tcp && ufw allow 80/tcp && ufw allow 443/tcp && ufw enable

adduser ddt && usermod -aG sudo ddt
# далее под ddt:

cd /home/ddt
git clone git@github.com:nonodoubt/ai_assistant_education.git bot
cd bot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install sentence-transformers
python services/db/build_db.py && python services/db/build_vectors.py
```

### systemd `/etc/systemd/system/ddt-bot.service`

```ini
[Unit]
Description=DDT Soyuz RAG chatbot
After=network.target

[Service]
Type=simple
User=ddt
WorkingDirectory=/home/ddt/bot
Environment="PATH=/home/ddt/bot/.venv/bin"
Environment="ADMIN_PASSWORD=ваш-пароль"
Environment="SECRET_KEY=ваш-секрет"
ExecStart=/home/ddt/bot/.venv/bin/python server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now ddt-bot
```

### Nginx `/etc/nginx/sites-available/ddt-bot`

```nginx
server {
    server_name bot.unionddt.ru;
    client_max_body_size 2M;

    location /admin {
        auth_basic "DDT Admin";
        auth_basic_user_file /etc/nginx/.htpasswd;
        proxy_pass http://127.0.0.1:8501;
        proxy_set_header Host $host;
        proxy_read_timeout 60s;
    }

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 60s;
    }

    listen 443 ssl;
    ssl_certificate /etc/letsencrypt/live/bot.unionddt.ru/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/bot.unionddt.ru/privkey.pem;
    include /etc/letsencrypt/options-ssl-nginx.conf;
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;
}
server {
    if ($host = bot.unionddt.ru) { return 301 https://$host$request_uri; }
    listen 80;
    server_name bot.unionddt.ru;
    return 404;
}
```

```bash
sudo apt install -y apache2-utils
sudo htpasswd -c /etc/nginx/.htpasswd admin   # Basic Auth для /admin
sudo certbot --nginx -d bot.unionddt.ru       # SSL
sudo nginx -t && sudo systemctl reload nginx
```

### Встраивание на сайт (WordPress)

WordPress → Code Snippets → Add Snippet → HTML Snippet → Site Wide Footer:

```html
<button id="ddt-bot-btn" style="position:fixed;bottom:80px;right:24px;width:64px;height:64px;
  border-radius:50%;background:linear-gradient(135deg,#005CA9,#003366);color:#fff;border:0;
  font-size:28px;cursor:pointer;box-shadow:0 4px 20px rgba(0,92,169,.4);z-index:9999">💬</button>

<iframe id="ddt-bot-frame" src="https://bot.unionddt.ru/?embed=1" title="Чат с помощником"
  style="position:fixed;bottom:152px;right:24px;width:483px;height:80vh;max-height:700px;
  border:0;border-radius:16px;box-shadow:0 8px 40px rgba(0,0,0,.2);z-index:9998;display:none"></iframe>

<script>
(function(){
  var btn=document.getElementById('ddt-bot-btn');
  var frame=document.getElementById('ddt-bot-frame');
  btn.addEventListener('click',function(){
    var open=frame.style.display==='block';
    frame.style.display=open?'none':'block';
    btn.innerHTML=open?'💬':'✕';
  });
  if(window.innerWidth<600){
    frame.style.width='calc(100vw - 16px)';
    frame.style.right='8px'; frame.style.left='8px';
    frame.style.height='calc(100vh - 112px)';
  }
})();
</script>
```

---

## Обновление данных

```bash
cp /путь/к/новому/programs.xlsx data/programs.xlsx
source .venv/bin/activate
python services/db/build_db.py      # валидация + сборка
python services/db/build_vectors.py # пересборка векторов
sudo systemctl restart ddt-bot
```

### Обновление кода

```bash
# Локально:
git add . && git commit -m "описание" && git push

# На сервере:
cd /home/ddt/bot
git stash && git pull && git stash pop
sudo systemctl restart ddt-bot
```

---

## Безопасность

| Слой | Что защищает |
|------|-------------|
| nginx Basic Auth | `/admin` — первый барьер (логин/пароль браузера) |
| Flask-сессия + `@admin_required` | все `/api/admin/*` — 401 без авторизации |
| ADMIN_PASSWORD в env | пароль формы не хранится в коде |
| HTTPS / Let's Encrypt | весь трафик зашифрован |

### Файлы вне git (.gitignore)

```
services/api_key_manager/api_accounts.xlsx
services/api_key_manager/key_stats.json
data/chat_logs.db
services/db/knowledge.db
```

---

## Команды управления

### Сервис

```bash
sudo systemctl status ddt-bot      # статус
sudo systemctl restart ddt-bot     # перезапуск (после кода или БД)
sudo journalctl -u ddt-bot -f      # логи в реальном времени
sudo journalctl -u ddt-bot -n 50   # последние 50 строк
```

### Nginx

```bash
sudo nginx -t                      # проверка конфига
sudo systemctl reload nginx        # применить изменения
```

### База данных

```bash
python services/db/build_db.py             # пересборка БД
python services/db/build_vectors.py        # пересборка векторов
python services/db/validate_programs.py    # только валидация

# Статистика БД:
python3 -c "
import sqlite3
c = sqlite3.connect('services/db/knowledge.db')
print('Программ:', c.execute('SELECT COUNT(*) FROM programs').fetchone()[0])
print('FAQ:', c.execute('SELECT COUNT(*) FROM faq').fetchone()[0])
print('Тегов:', c.execute('SELECT COUNT(*) FROM program_tags').fetchone()[0])
"
```

### Обновление зависимостей

```bash
# Сгенерировать актуальный requirements.txt:
pip freeze | grep -v "@ file://" > requirements.txt
```
