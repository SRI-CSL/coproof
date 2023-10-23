"""CoProver: The Cueology of Proof

Description:
  Utility methods all in one place!

@copyright: SRI International, 2022
"""

import os
import re

from typing import List


def path_exists(input_path: str) -> bool:
    return os.path.exists(input_path)


def is_file(input_path: str) -> bool:
    return os.path.isfile(input_path)


def is_directory(input_path: str) -> bool:
    return os.path.isdir(input_path)


def get_file_paths(input_dir: str, ends_with: str = '') -> List:
    """Obtain a list composed with the paths to all the files in a directory

    Args:
      input_dir: path to input directory
      ends_with: filter to select specific files within a directory, default: ''

    Returns:
      List containing the paths to all desired files within a directory
    """
    if ends_with != '':
        file_paths = [f for f in os.listdir(input_dir) if is_file(
            os.path.join(input_dir, f)) and f.endswith(ends_with)]
    else:
        file_paths = [f for f in os.listdir(input_dir) if is_file(os.path.join(input_dir, f))]

    return file_paths


def get_numbers_from_str(input_str: str) -> List:
    result = list(map(int, re.findall(r'\d+', input_str)))
    return result


def read_file_lines(input_file: str) -> List:
    res = []

    with open(input_file) as f:
        res = [f_line.rstrip() for f_line in f]

    return res
