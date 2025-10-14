# mook_analyzer.py
"""
MOOK BASE: HTTP Server + Logging = Observabilidad Mínima
Pattern detectado: Servidor básico generando logs automáticos
"""

import re
from datetime import datetime

class MookTrafficAnalyzer:
    def __init__(self):
        self.mook_patterns = {
            'basic_requests': r'(\d+\.\d+\.\d+\.\d+).*"GET (.*?) HTTP',
            'status_codes': r'" (\d{3}) ',
            'file_access': r'GET (/[\w\.-]*)',
            'timestamps': r'\[(.*?)\]'
        }
        self.mook_insights = []
    
    def analyze_mook_log(self, log_line):
        """Extraer conocimiento operacional mínimo del log"""
        insights = {'timestamp': datetime.now(), 'raw': log_line}
        
        # 1. IP y recurso (qué está accediendo)
        ip_match = re.search(self.mook_patterns['basic_requests'], log_line)
        if ip_match:
            insights['ip'] = ip_match.group(1)
            insights['resource'] = ip_match.group(2)
            insights['type'] = 'request'
        
        # 2. Status code (cómo respondió)
        status_match = re.search(self.mook_patterns['status_codes'], log_line)
        if status_match:
            insights['status'] = int(status_match.group(1))
            insights['health'] = 'healthy' if insights['status'] == 200 else 'issue'
        
        # 3. Timestamp (cuándo)
        time_match = re.search(self.mook_patterns['timestamps'], log_line)
        if time_match:
            insights['server_time'] = time_match.group(1)
        
        self.mook_insights.append(insights)
        return insights

# MOOK VALIDATION - Tu setup actual
mook = MookTrafficAnalyzer()

# Analizar tus logs existentes
test_logs = [
    '127.0.0.1 - - [03/Oct/2025 17:47:05] "GET / HTTP/1.1" 200 -',
    '127.0.0.1 - - [03/Oct/2025 17:47:06] "GET /favicon.ico HTTP/1.1" 404 -'
]

print("🔍 ANALIZANDO MOOK BASE:")
for log in test_logs:
    insight = mook.analyze_mook_log(log)
    print(f"📄 {insight}")

print(f"\n✅ MOOK CONFIRMADO: {len(mook.mook_insights)} insights operacionales")
