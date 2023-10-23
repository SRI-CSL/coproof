"""CoProver: The Cueology of Proof

Description:
  Data processing for MetiTarski variable ordering problem,
  accomodating transformer architectures.

@copyright: SRI International, 2022
"""

import os
import sys
import time
import argparse
import pandas as pd

from utils import path_exists, is_directory, is_file, get_file_paths, get_numbers_from_str, read_file_lines


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data_dir", type=str, required=True,
                        help="path to input data files")

    parser.add_argument("-l", "--label_dir", type=str, required=True,
                        help="path to label data files")

    parser.add_argument("-o", "--output_file", type=str, required=True,
                        help="path to output file")

    args = parser.parse_args()
    return args


def get_label(input_file: str) -> int:
    variable_combinations = [
        'x1\tx2\tx3',
        'x1\tx3\tx2',
        'x2\tx1\tx3',
        'x2\tx3\tx1',
        'x3\tx1\tx2',
        'x3\tx2\tx1']

    res = []
    with open(input_file) as f:
        res = [f_line.rstrip() for f_line in f]

    exec_times = []

    for i, j in enumerate(res):
        if j in variable_combinations:
            if not res[i+1]:
                exec_times.append(100000)
            else:
                exec_times.append(float(res[i+1]))

    min_idx = exec_times.index(min(exec_times))

    return min_idx


def main():
    """Execute the preprocessing step"""

    start_time = time.time()
    config = parse_arguments()

    if path_exists(input_path=config.data_dir) and is_directory(input_path=config.data_dir):
        path_input_data = config.data_dir
    else:
        raise NotADirectoryError(f"{config.data_dir} is not a directory.")

    if path_exists(input_path=config.label_dir) and is_directory(input_path=config.label_dir):
        path_label_data = config.label_dir
    else:
        raise NotADirectoryError(f"{config.label_dir} is not a directory.")

    path_output_file = config.output_file

    print(f"- Input: {path_input_data}\n- Labels: {path_label_data}")

    input_file_paths = get_file_paths(input_dir=path_input_data, ends_with=".ml")
    print(f"- There are {len(input_file_paths)}-input files ending with .ml.")

    result_data = []

    for i, file_name in enumerate(input_file_paths):
        f_number = get_numbers_from_str(input_str=file_name)
        l_file = f"comp_times{f_number[0]}.txt"

        if path_exists(input_path=os.path.join(path_input_data, file_name)) and path_exists(input_path=os.path.join(path_label_data, l_file)):
            # 1. extracting all polynomials from a given input file
            polynomial_list = read_file_lines(
                input_file=os.path.join(path_input_data, file_name))[2:]

            poly_sequence_data = ' '.join(polynomial_list)
            poly_sequence_label = get_label(input_file=os.path.join(path_label_data, l_file))
            # print(f"sequence: {poly_sequence_data}\tlabel: {poly_sequence_label}")

            result_data.append({"source_text": poly_sequence_data,
                               "target_text": str(poly_sequence_label)})

            if i % 1000 == 0:
                print(f"-- Processed {i} files!")

    print(f"- Output contains: {len(result_data)} processed polynomials.")

    # save results to .csv file
    df_output_dataset = pd.DataFrame(result_data)
    df_output_dataset.to_csv(path_output_file, sep='\t')

    print(f"- Output file saved at: {path_output_file}")
    print(f"- Execution completed in {time.time() - start_time} seconds.")
    return


if __name__ == '__main__':
    main()
