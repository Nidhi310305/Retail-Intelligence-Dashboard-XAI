from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_dataset(uploaded_file: Any) -> pd.DataFrame:
    """Load CSV or Excel input into a DataFrame."""
    if uploaded_file is None:
        raise ValueError("No file provided")

    file_name = getattr(uploaded_file, "name", "") or ""
    suffix = Path(file_name).suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(uploaded_file, encoding="latin1")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file type. Upload a CSV or Excel file.")


def load_sample_dataset(path: str | Path) -> pd.DataFrame:
    """Load the bundled sample dataset for local testing."""
    sample_path = Path(path)
    return pd.read_csv(sample_path, encoding="latin1")
