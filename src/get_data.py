import pandas as pd
from pathlib import Path

test_input_path = Path(__file__).parent.parent / 'data' / \
    'greenwashing' / 'greenwashing_test_input.csv'


def load_test_data(rows: int = None) -> pd.DataFrame:
    """
    Load test data from a CSV file.

    Args:
        rows (int, optional): Number of rows to load. Defaults to None (load all rows).

    Returns:
        pd.DataFrame: DataFrame containing the test data.
    """
    if not test_input_path.exists():
        raise FileNotFoundError(f"File not found: {test_input_path}")

    df = pd.read_csv(test_input_path, encoding='utf-8')
    return df.head(rows) if rows else df
