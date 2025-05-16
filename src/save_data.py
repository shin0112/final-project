import pandas as pd
from pathlib import Path
from datetime import datetime
import logging


def save_results(results, test_input_df=None, filename="model_output"):
    """
    결과 리스트를 CSV 파일로 저장하는 함수.

    Args:
        results (list[dict]): {"article": ..., "context": ..., "answer": ...} 형식 리스트
        filename (str): 저장할 파일 이름
    """
    if not results:
        logging.warning("⚠ 저장할 결과가 비어 있습니다. 파일을 생성하지 않습니다.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename}_{timestamp}.csv"

    output_path = Path(__file__).parent.parent / "results" / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_result = pd.DataFrame(results)

    if test_input_df is not None and "full_text" in test_input_df.columns:
        df_result["original_text"] = test_input_df["full_text"].values[:len(
            df_result)]
    if test_input_df is not None and "greenwashing_level" in test_input_df.columns:
        df_result["label"] = test_input_df["greenwashing_level"].values[:len(
            df_result)]

    df_result.to_csv(output_path, index=False, encoding='utf-8-sig')
    logging.info(f"[결과 저장 완료] {output_path}")
