import pandas as pd

from fast_ml_preprocessing.utils.filesystem_utils import get_file_extension, is_csv, is_excel


class DatasetReadingOperation:
    def read(self, filepath: str, index_col: int, separator: str) -> pd.DataFrame:
        index_col = None if index_col == 0 else 0
        file_extension: str = get_file_extension(filepath)
        if is_csv(file_extension):
            return pd.read_csv(
                filepath,
                header=0,
                index_col=index_col,
                on_bad_lines='skip',
                delimiter=separator
            )
        if is_excel(file_extension):
            return pd.read_excel(
                filepath,
                header=0,
                index_col=index_col,
                sheet_name=0
            )
        raise ValueError('Invalid file format')
