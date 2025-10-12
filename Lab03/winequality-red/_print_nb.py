import json
from pathlib import Path
nb = json.loads(Path("winequality-red.ipynb").read_text(encoding="utf-8"))
for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") == "code":
        print(f"\n# Cell {i}\n")
        print(''.join(cell.get("source", [])))
