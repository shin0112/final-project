import os
import logging
import pandas as pd
from pathlib import Path

import query_processor
import process_data

test_input_path = Path(__file__).parent.parent / 'data' / \
    'greenwashing' / 'test_data.csv'
test_input_c_path = Path(__file__).parent.parent / 'data' / \
    'greenwashing' / 'test_data_compressed.csv'
final_test_path = Path(__file__).parent.parent / 'data' / \
    'greenwashing' / 'final_test_data.csv'


def load_test_data(rows: int = None) -> pd.DataFrame:
    """
    Load test data from a CSV file.

    Args:
        rows (int, optional): Number of rows to load. Defaults to None (load all rows).

    Returns:
        pd.DataFrame: DataFrame containing the test data.
    """
    if not os.path.exists(final_test_path):
        process_data.compress_article(test_input_path, test_input_c_path)
        df = pd.read_csv(test_input_c_path, encoding='utf-8')

        df_0 = df[df['greenwashing_level'] ==
                  0.0].sample(n=12, random_state=42)
        df_1 = df[df['greenwashing_level'] ==
                  1.0].sample(n=12, random_state=42)
        df_05 = df[df['greenwashing_level'] ==
                   0.5].sample(n=6, random_state=42)

        final_df = pd.concat([df_0, df_1, df_05]).reset_index(drop=True)
        final_df.to_csv(final_test_path, index=False)

    df = pd.read_csv(final_test_path, encoding='utf-8')
    logging.info(f"[테스트 데이터 로드] {len(df)}개 문장 로드")
    return df.head(rows) if rows else df


def load_data(legalize: bool = True):
    test_input = load_test_data()
    test_input = query_processor.preprocess_articles(test_input, legalize)

    if not test_input.empty:
        logging.info(
            f"[확인] {test_input['compressed_article'].iloc[0][:1000]} {test_input['ct'].iloc[0]}")
    else:
        logging.warning("⚠ 처리된 문서가 없습니다!")

    return test_input
