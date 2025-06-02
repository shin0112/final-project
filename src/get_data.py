import logging
import pandas as pd
from pathlib import Path

import query_processor
import process_data

test_input_path = Path(__file__).parent.parent / 'data' / \
    'greenwashing' / 'test_data.csv'
test_input_c_path = Path(__file__).parent.parent / 'data' / \
    'greenwashing' / 'test_data_compressed.csv'


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
    test_input = load_test_data()

    # 무작위 샘플 40개 뽑기
    # test_input = test_input.sample(n=40).reset_index(drop=True)

    # 그린워싱 없음 비중 향상
    negative_samples = test_input[test_input["greenwashing_level"] == 0]
    selected_negatives = negative_samples.sample(
        n=min(30, len(negative_samples)), random_state=42)

    remaining = 30 - len(selected_negatives)
    if remaining > 0:
        remaining_samples = test_input.drop(
            selected_negatives.index).sample(n=remaining, random_state=42)
        test_input = pd.concat(
            [selected_negatives, remaining_samples], ignore_index=True)
    else:
        test_input = selected_negatives.reset_index(drop=True)

    logging.info(f"[테스트 데이터 로드] {len(test_input)}개 문장 로드")

    test_input = query_processor.preprocess_articles(test_input, legalize)

    if not test_input.empty:
        logging.info(
            f"[확인] {test_input['compressed_article'].iloc[0][:1000]} {test_input['ct'].iloc[0]}")
    else:
        logging.warning("⚠ 처리된 문서가 없습니다!")

    return test_input
