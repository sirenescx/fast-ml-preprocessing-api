import pathlib


def get_file_extension(filepath: str) -> str:
    return pathlib.Path(filepath).suffix


def is_csv(extension: str) -> bool:
    return extension == '.csv'


def is_excel(extension: str) -> bool:
    return extension in ('.xls', '.xlsx')

