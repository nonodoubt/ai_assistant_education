#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT=8501

echo "════════════════════════════════════════════════"
echo "  ДДТ «Союз» — RAG чат-бот (Flask + Vue.js)"
echo "════════════════════════════════════════════════"

# Установка Flask если нет
pip install flask --quiet --break-system-packages 2>/dev/null || \
pip install flask --quiet 2>/dev/null

# Освобождаем порт
if lsof -i :$PORT > /dev/null 2>&1; then
    echo "Порт $PORT занят, освобождаю..."
    kill -9 $(lsof -t -i:$PORT) 2>/dev/null || true
    for i in $(seq 1 10); do
        if ! lsof -i :$PORT > /dev/null 2>&1; then break; fi
        sleep 1
    done
fi

echo ""
echo "Запускаю на http://localhost:$PORT"
echo "════════════════════════════════════════════════"

cd "$SCRIPT_DIR"
python3 server.py
