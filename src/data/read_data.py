import os
from pathlib import Path

import pandas as pd


def root_dir():
    return Path(os.path.abspath(os.path.dirname(__file__))).parent.parent


if __name__ == '__main__':
    pd.set_option('display.max_columns', 10)
