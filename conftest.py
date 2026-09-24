import sys
from pathlib import Path

# Добавляем корень проекта в sys.path, чтобы pytest видел модуль src
sys.path.insert(0, str(Path(__file__).parent))