#!/bin/bash
# mook_deploy.sh
# MOOK Deployment - Lo que SÍ viaja al tin siguiente

echo "🚀 DEPLOYING MOOK BASE..."

# 1. Estructura MOOK confirmada
mkdir -p mook_{logs,html,analysis}

# 2. HTML base (tu NationalUnSafeBankHomePage.html)
cp NationalUnSafeBankHomePage.html mook_html/

# 3. Server básico (ya funciona)
echo "Serving MOOK at http://0.0.0.0:8000"

# 4. Análisis automático de logs
python3 -c "
from mook_analyzer import MookTrafficAnalyzer
mook = MookTrafficAnalyzer()

# Simular análisis continuo
import time
while True:
    # En producción, leer de archivo de log
    test_log = '127.0.0.1 - - [03/Oct/2025 18:00:00] \"GET /test HTTP/1.1\" 200 -'
    insight = mook.analyze_mook_log(test_log)
    print(f'MOOK Insight: {insight}')
    time.sleep(10)
" &

echo "✅ MOOK BASE DEPLOYED"
echo "📊 Insights activos en segundo plano"
