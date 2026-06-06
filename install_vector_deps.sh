#!/bin/bash
# install_vector_deps.sh — Установка зависимостей для гибридного поиска
#
# Запуск: chmod +x install_vector_deps.sh && ./install_vector_deps.sh

set -e

echo "═══════════════════════════════════════════════"
echo "  Установка зависимостей для векторного поиска"
echo "═══════════════════════════════════════════════"

echo ""
echo "1. Основные пакеты..."
pip install streamlit pandas openpyxl python-docx requests

echo ""
echo "2. sqlite-vec (векторное хранилище)..."
pip install sqlite-vec

echo ""
echo "3. sentence-transformers + torch (модели эмбеддингов и реранкера)..."
pip install sentence-transformers torch

echo ""
echo "4. Загрузка моделей (первый запуск будет дольше)..."
python3 -c "
from sentence_transformers import SentenceTransformer, CrossEncoder
print('Загрузка bge-m3...')
SentenceTransformer('BAAI/bge-m3')
print('✅ bge-m3 загружен')
print('Загрузка bge-reranker-v2-m3...')
CrossEncoder('BAAI/bge-reranker-v2-m3')
print('✅ bge-reranker-v2-m3 загружен')
"

echo ""
echo "5. Сборка векторной БД..."
cd "$(dirname "$0")/services/db"
python3 build_vectors.py

echo ""
echo "═══════════════════════════════════════════════"
echo "  ✅ Готово! Запускайте: ./run.sh"
echo "═══════════════════════════════════════════════"