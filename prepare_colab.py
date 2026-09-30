"""Restore and verify the bundled interim Parquet panel for local use."""
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "data/interim/materials_equity_panel.parquet"
PARTS = [TARGET.with_name(TARGET.name + f".part{i:02d}") for i in (1, 2)]
EXPECTED_SHA256 = "edbf25dda3f51b5bc06631cebe1b461b4b169822951a6829dd3b65bffbeeb27c"

if not all(part.is_file() for part in PARTS):
    raise FileNotFoundError("Both interim panel parts must be present in data/interim/.")

checksum = sha256()
for part in PARTS:
    with part.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(chunk)
if checksum.hexdigest() != EXPECTED_SHA256:
    raise ValueError("Interim panel parts failed checksum verification.")

if TARGET.exists():
    existing = sha256(TARGET.read_bytes()).hexdigest()
    if existing == EXPECTED_SHA256:
        print(f"Already restored: {TARGET}")
        raise SystemExit(0)

with TARGET.open("wb") as output:
    for part in PARTS:
        with part.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                output.write(chunk)
print(f"Restored and verified: {TARGET}")
