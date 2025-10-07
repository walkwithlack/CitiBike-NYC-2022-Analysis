from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "2.Data" / "Prepared Data"

def csv_path(name: str) -> Path:
    p = DATA_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"Expected CSV at: {p}\n"
                                f"→ Ensure this file is committed to the repo.")
    return p
