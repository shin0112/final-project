from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
import re


def save_results(results, test_input_df=None, filename="model_output"):
    """
    결과 리스트를 CSV 파일로 저장하는 함수.
    """
    if not results:
        logging.warning("⚠ 저장할 결과가 비어 있습니다. 파일을 생성하지 않습니다.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename}_{timestamp}.csv"

    output_path = Path(__file__).parent.parent / "results" / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_result = pd.DataFrame(results)

    # 원본 기사 및 라벨 복사
    if test_input_df is not None:
        if "full_text" in test_input_df.columns:
            df_result["original_text"] = test_input_df["full_text"].values[:len(
                df_result)]
        if "greenwashing_level" in test_input_df.columns:
            df_result["label"] = test_input_df["greenwashing_level"].values[:len(
                df_result)]

    # 모델 응답 파싱 및 평가
    parsed_list = [parse_model_answer(r["answer"]) for r in results]
    df_parsed = pd.DataFrame(parsed_list)
    df_result = pd.concat([df_result, df_parsed], axis=1)
    if "label" in df_result.columns:
        df_result["is_correct"] = df_result.apply(
            is_prediction_correct, axis=1)

    df_result.to_csv(output_path, index=False, encoding='utf-8-sig')
    logging.info(f"[결과 저장 완료] {output_path.resolve()}")


def is_prediction_correct(row, label_col="label") -> bool | None:
    if not row["valid_format"] or row["judgement"] is None:
        return None

    judgement = row["judgement"]
    label = row[label_col]

    if label == 1 and "없음" in judgement:
        return False
    if label == 0 and "있음" in judgement:
        return False
    return True


def parse_model_answer(answer: str) -> dict:
    """
    모델의 답변에서 판단/근거/해결방안을 정규식으로 추출
    """
    try:
        판단 = re.search(r"\s*판단\s*:\s*(.+?)\n", answer)
        근거 = re.search(r"\s*근거\s*:\s*(.+?)\n", answer)
        법률 = re.search(r"\s*법률\s*:\s*(.+?)\n", answer)
        해결 = re.search(r"\s*해결방안\s*:\s*(.+)", answer)

        return {
            "judgement": 판단.group(1).strip() if 판단 else "",
            "reason": 근거.group(1).strip() if 근거 else "",
            "law": 법률.group(1).strip() if 법률 else "",
            "suggestion": 해결.group(1).strip() if 해결 else "",
            "valid_format": 판단 is not None
        }

    except Exception:
        return {
            "judgement": None,
            "reason": None,
            "law": None,
            "suggestion": None,
            "valid_format": False
        }


def is_prediction_correct_fixed(row, label_col="label") -> bool | None:
    if not row["valid_format"] or row["judgement"] is None:
        return None

    judgement = row["judgement"]
    label = row[label_col]

    if not isinstance(judgement, str):
        return None

    if label == 1 and "없음" in judgement:
        return False
    if label == 0 and ("있음" in judgement or "의심" in judgement):
        return False
    return True


def evaluate_model(df: pd.DataFrame, label_col="label", judgement_col="judgement") -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered_df = df[df[label_col] != 0.5].copy()
    filtered_df[label_col] = filtered_df[label_col].astype(int)

    filtered_df["judgement_bin"] = filtered_df[judgement_col].apply(
        lambda x: 1 if isinstance(x, str) and ("있음" in x or "의심" in x) else 0
    )

    y_true = filtered_df[label_col]
    y_pred = filtered_df["judgement_bin"]

    report = classification_report(y_true, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_true, y_pred)

    report_df = pd.DataFrame(report).transpose()
    conf_matrix_df = pd.DataFrame(conf_matrix,
                                  index=["실제 없음", "실제 있음"],
                                  columns=["예측 없음", "예측 있음"])
    return report_df, conf_matrix_df


def save_and_evaluate_results(
    results,
    test_input_df=None,
    filename="model_output"
):
    if not results:
        logging.warning("⚠ 저장할 결과가 비어 있습니다. 파일을 생성하지 않습니다.")
        return None, None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename}_{timestamp}.csv"

    output_path = Path(__file__).parent.parent / "results" / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_result = pd.DataFrame(results)

    if test_input_df is not None:
        if "full_text" in test_input_df.columns:
            df_result["original_text"] = test_input_df["full_text"].values[:len(
                df_result)]
        if "greenwashing_level" in test_input_df.columns:
            df_result["label"] = test_input_df["greenwashing_level"].values[:len(
                df_result)]

    parsed_list = [parse_model_answer(r["answer"]) for r in results]
    df_parsed = pd.DataFrame(parsed_list)
    df_result = pd.concat([df_result, df_parsed], axis=1)

    if "label" in df_result.columns:
        df_result["is_correct"] = df_result.apply(
            is_prediction_correct_fixed, axis=1)

    df_result["judgement_bin"] = df_result["judgement"].apply(
        lambda x: 1 if isinstance(x, str) and ("있음" in x or "의심" in x) else 0
    )

    df_result.to_csv(output_path, index=False, encoding='utf-8-sig')
    logging.info(f"[결과 저장 완료] {output_path.resolve()}")

    report_df, conf_matrix_df = evaluate_model(
        df_result, label_col="label", judgement_col="judgement")
    return report_df, conf_matrix_df
