#!/bin/bash
# НЕ используем set -e — clo может возвращать ненулевой код

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT=8501

echo "════════════════════════════════════════════════"
echo "  ДДТ «Союз» — RAG чат-бот"
echo "════════════════════════════════════════════════"

if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 не найден"
    exit 1
fi

echo "Проверяю зависимости..."
pip install streamlit pandas openpyxl python-docx requests --quiet --break-system-packages 2>/dev/null || \
pip install streamlit pandas openpyxl python-docx requests --quiet 2>/dev/null

if [ ! -f "$SCRIPT_DIR/services/db/knowledge.db" ]; then
    echo "База данных не найдена. Создаю..."
    cd "$SCRIPT_DIR"
    python3 services/db/build_db.py
fi

if [ "$1" = "--cloudpub" ]; then
    echo ""

    # Освобождаем порт
    if lsof -i :$PORT > /dev/null 2>&1; then
        echo "Порт $PORT занят, освобождаю..."
        kill -9 $(lsof -t -i:$PORT) 2>/dev/null || true
        for i in $(seq 1 10); do
            if ! lsof -i :$PORT > /dev/null 2>&1; then
                echo "Порт $PORT свободен."
                break
            fi
            echo "  ждём... ($i)"
            sleep 1
        done
    fi

    # Запускаем Streamlit в фоне
    echo "Запускаю Streamlit на порту $PORT..."
    cd "$SCRIPT_DIR"
    streamlit run app.py \
        --server.port $PORT \
        --server.address 0.0.0.0 \
        --server.headless true \
        --browser.gatherUsageStats false \
        &
    STREAMLIT_PID=$!

    # Ждём пока Streamlit стартует
    echo "Ожидаю запуск Streamlit..."
    for i in $(seq 1 15); do
        if curl -s http://localhost:$PORT > /dev/null 2>&1; then
            echo "Streamlit запущен."
            break
        fi
        sleep 1
    done

    echo ""
    echo "Запускаю CloudPub..."
    echo "════════════════════════════════════════════════"

    if ! command -v clo &> /dev/null; then
        echo "❌ clo не найден"
        kill $STREAMLIT_PID 2>/dev/null
        exit 1
    fi

    if ! clo ls 2>/dev/null | grep -q "$PORT"; then
        echo "Регистрирую сервис..."
        clo publish http 127.0.0.1:$PORT || true
    fi

    # Обработка Ctrl+C
    cleanup() {
        echo ""
        echo "Останавливаю..."
        kill $STREAMLIT_PID 2>/dev/null
        echo "Остановлено."
        exit 0
    }
    trap cleanup INT TERM

    # clo start в foreground — держит туннель открытым
    echo "Запускаю туннель (Ctrl+C для остановки)..."
    clo start 36ba4f1e-2e57-432d-ad8a-21b5297ff213 || true

    # Если clo завершился сам — ждём Streamlit
    echo ""
    echo "CloudPub завершился. Streamlit продолжает работать на http://localhost:$PORT"
    echo "Нажмите Ctrl+C для полной остановки."
    wait $STREAMLIT_PID 2>/dev/null

else
    echo ""
    echo "Запускаю Streamlit на http://localhost:$PORT"
    echo "(Для публичного URL: ./run.sh --cloudpub)"
    echo "════════════════════════════════════════════════"
    echo ""

    cd "$SCRIPT_DIR"
    streamlit run app.py \
        --server.port $PORT \
        --server.address 0.0.0.0 \
        --server.headless true \
        --browser.gatherUsageStats false
fi