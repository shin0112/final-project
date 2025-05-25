import logging
import pandas as pd
from pathlib import Path

import query_processor
import process_data

test_input_path = Path(__file__).parent.parent / 'data' / \
    'greenwashing' / 'greenwashing_test.csv'
test_input_c_path = Path(__file__).parent.parent / 'data' / \
    'greenwashing' / 'greenwashing_test_compressed.csv'


def load_test_data(rows: int = None) -> pd.DataFrame:
    """
    Load test data from a CSV file.

    Args:
        rows (int, optional): Number of rows to load. Defaults to None (load all rows).

    Returns:
        pd.DataFrame: DataFrame containing the test data.
    """
    process_data.compress_article(test_input_path, test_input_c_path)

    df = pd.read_csv(test_input_c_path, encoding='utf-8')
    return df.head(rows) if rows else df


def load_data(legalize: bool = True):
    # get input data 10개
    test_input = load_test_data(3)
    test_input = test_input.dropna(subset=["compressed_article"])

    logging.info(f"[테스트 데이터 로드] {len(test_input)}개 문장 로드")

    test_input = query_processor.preprocess_articles(test_input, legalize)

    if not test_input.empty:
        logging.info(f"[확인] {test_input['compressed_article'].iloc[0][:1000]}")
    else:
        logging.warning("⚠ 처리된 문서가 없습니다!")

    return test_input
