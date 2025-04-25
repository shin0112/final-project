import json
import re
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

json_path = Path(__file__).parent.parent / 'config' / 'law_file_paths.json'


def get_ko_law():
    """
    This function is used to get the list of law file paths.    
    """
    if not json_path.exists():
        raise FileNotFoundError(f"File not found: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    path_list = data['ko_law_path_list']
    return docs_load(path_list)


def get_ko_guideline():
    """
    This function is used to get the list of guideline file paths.    
    """
    if not json_path.exists():
        raise FileNotFoundError(f"File not found: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    path_list = data['ko_guideline_path_list']
    return docs_load(path_list)


def docs_load(path_list: list) -> list:
    """
    Loads the PDF files from the specified paths and processes them.

    Args:
        path_list (list): A list of file paths for the PDF files to load.
    """
    base_path = Path(__file__).parent.parent
    pdf_list = []  # List to store loaded PDF data

    # todo : pdf 파일인지 확인하고, 아니면 web data load 하기 + text encoding
    for pdf_path in path_list:
        full_pdf_path = base_path / pdf_path
        loader = PyPDFLoader(str(full_pdf_path))
        data = loader.load()
        print(f"Loaded {len(data)} pages from {full_pdf_path}")

        cleaned_data = []
        for page in data:
            page.page_content = normalize_newline(page.page_content)
            page.page_content = clean_text(page.page_content)
            cleaned_data.append(page)

        pdf_list.append(cleaned_data)

    return pdf_list


def normalize_newline(text: str) -> str:
    # \r 제거
    text = text.replace('r', '')
    # 2개 이상 \n 하나로 통합
    text = re.sub(r'\n{2,}', '\n\n', text)
    # 단일 줄바꿈 → 공백
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    return text.strip()


def clean_text(text: str) -> str:
    # 날짜 패턴 삭제 (ex. 2023.05.14. / 2021년 12월 31일)
    text = re.sub(r'\d{4}[년.\-\/]\d{1,2}[월.\-\/]\d{1,2}[일.]?', '', text)

    # 페이지 번호 삭제 (ex. - 5 -, Page 2 of 10)
    text = re.sub(r'-\s*\d+\s*-', '', text)
    text = re.sub(r'Page\s*\d+\s*of\s*\d+', '', text)

    # 불필요한 포맷 잡음 삭제
    text = re.sub(r'[-=]{3,}', '', text)  # --- === 같이 반복된 줄 삭제
    text = re.sub(r'\(끝\)', '', text)

    # 특정 의미 없는 단어 삭제
    text = re.sub(r'작성자\s*:.*', '', text)
    text = re.sub(r'검토자\s*:.*', '', text)
    text = re.sub(r'목차', '', text)

    # 연속 공백 정리
    text = re.sub(r'\s+', ' ', text)

    return text.strip()
