import sys
from pathlib import Path

# Verify local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import adjointrwm
print("Local adjointrwm imported successfully:", adjointrwm.__file__)
