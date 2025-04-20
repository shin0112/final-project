import json
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


def get_ko_law():
    """
    This function is used to get the list of law file paths.    
    """
    json_path = Path(__file__).parent.parent / 'config' / 'law_file_paths.json'

    if not json_path.exists():
        raise FileNotFoundError(f"File not found: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    path_list = data['ko_law_path_list']
    return docs_load(path_list)


def docs_load(path_list: list) -> list:
    """
    Loads the PDF files from the specified paths and processes them.

    Args:
        path_list (list): A list of file paths for the PDF files to load.
    """
    pdf_list = []  # List to store loaded PDF data

    # todo : pdf 파일인지 확인하고, 아니면 web data load 하기 + text encoding
    for pdf_path in path_list:
        loader = PyPDFLoader(pdf_path)
        data = loader.load()
        print(f"Loaded {len(data)} pages from {pdf_path}")
        pdf_list.append(data)

    return pdf_list
