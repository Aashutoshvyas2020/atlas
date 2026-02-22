from __future__ import annotations

import json
import os
import pathlib
from datetime import datetime
from typing import Any


def log(message: str, **fields: Any) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    if fields:
        print(f"[{ts}] {message} | {json.dumps(fields, ensure_ascii=True)}")
    else:
        print(f"[{ts}] {message}")


def load_dotenv_file(path: str = ".env") -> None:
    p = pathlib.Path(path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip().lstrip("\ufeff")
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip().lstrip("\ufeff")
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v
