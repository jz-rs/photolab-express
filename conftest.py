# Hace que `import src` funcione al correr `pytest` desde la raíz del repositorio.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
