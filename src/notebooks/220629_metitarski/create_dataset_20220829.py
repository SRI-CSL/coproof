"""CoProver: The Cueology of Proof

@description:
  Data processing for MetiTarski variable ordering problem.
  This updated file revisits some of the methods used and clarifies the preprocessing steps.

@copyright: SRI International, 2022
"""

import os
import sys

import re
import time
import argparse
import numpy as np
import pandas as pd
import json  # only for printing the dictionary, to be removed

from os import listdir
from os.path import isfile, join
from typing import List, Any, Dict
from datetime import datetime

from utils import path_exists, is_directory, is_file, get_file_paths, get_numbers_from_str, read_file_lines


def get_polynomials(input_file: str) -> List:
    res = []

    with open(input_file) as f:
        res = [f_line.rstrip() for f_line in f]

    return res


def get_polynomials_test(input_text: str) -> List:
    # p = input_text.split('\n').rstrip()[2:]
    p = [i.rstrip() for i in input_text.split('\n')]
    return p


def get_numbers_from_str(input_str: str) -> List:
    result = list(map(int, re.findall(r'\d+', input_str)))
    return result


def get_power(input_str: str, target_variable: str) -> int:
    res = 1

    a = input_str.split(target_variable)
    if a[1].startswith('^'):
        p = get_numbers_from_str(input_str=a[1])
        res = p[0]

    return res


def get_variable_degree(input_polynomial: str, target_variable: str) -> List:
    # account for more than one instance of x1, x2, x3 in the polynomial
    res = []

    occurrences_v = [v.start() for v in re.finditer(target_variable, input_polynomial)]

    if len(occurrences_v) == 0:
        # no occurrence of target found
        res.append(0)
    elif len(occurrences_v) == 1:
        # only one occurrence of target found
        p_value = get_power(input_str=input_polynomial, target_variable=target_variable)
        res.append(p_value)
    else:
        # more than one occurrence of target found
        for i in range(len(occurrences_v)):
            if i == len(occurrences_v) - 1:
                p_value = get_power(
                    input_str=input_polynomial[occurrences_v[i]:], target_variable=target_variable)
                res.append(p_value)
            else:
                p_value = get_power(
                    input_str=input_polynomial[occurrences_v[i]:occurrences_v[i+1]], target_variable=target_variable)
                res.append(p_value)

    return res


def process_polynomial(input_polynomial: str) -> Dict[str, int]:
    # if 0 the variable does not exist
    result = {'x1': None, 'x2': None, 'x3': None}

    result['x1'] = get_variable_degree(input_polynomial=input_polynomial, target_variable='x1')
    result['x2'] = get_variable_degree(input_polynomial=input_polynomial, target_variable='x2')
    result['x3'] = get_variable_degree(input_polynomial=input_polynomial, target_variable='x3')

    return result


def get_number_of_monomials(processed_pol: Dict) -> int:

    cnt_x1 = 0
    cnt_x2 = 0
    cnt_x3 = 0

    for p in processed_pol:
        cnt_x1 += sum(1 for i in p['x1'] if i != 0)
        cnt_x2 += sum(1 for i in p['x2'] if i != 0)
        cnt_x3 += sum(1 for i in p['x3'] if i != 0)

    return cnt_x1, cnt_x2, cnt_x3


def get_proportions(processed_pol: Dict, target_variable: str) -> float:
    cnt_target_occurrences = 0

    for p in processed_pol:
        # check whether target variable existed in p-polynomial
        s_v = sum(1 for i in p[target_variable] if i != 0)
        if s_v != 0:
            cnt_target_occurrences += 1

    # print(target_variable, cnt_target_occurrences, len(processed_pol))
    return cnt_target_occurrences/len(processed_pol) if len(processed_pol) > 0 else 0


def get_number_of_terms(polynomial_list: List) -> int:
    """Obtain the total number of terms in all polynomials

    Args:
      polynomial list: the list of polynomials provided in a .ml file

    Returns:
      the sum of number of terms in all polynomials
    """

    total_no_terms = sum(len(polynomial.split("_")) for polynomial in polynomial_list)

    return total_no_terms


def get_no_terms_per_variable(input_polynomial: str, target_variable: str) -> int:
    """Huang et al. 2019: The proportion of a variable occurring in monomials
       is the number of terms containing the variable divided by
       total number of terms in all polynomials.

    Args:
      input_polynomial: polynomial to process in .ml format
      target_variable: variable to search for in polynomial terms

    Returns:
      Number of terms containing the target variable
    """

    polynomial_terms = input_polynomial.split("_")

    cnt_target = 0
    for i in polynomial_terms:
        if target_variable in i:
            cnt_target += 1

    return cnt_target


def get_total_no_terms_per_variable(polynomial_list: List, target_variable: str) -> int:
    """Obtain the total number of terms containing a target variable in all
       available polynomials.

    Args:
      polynomial_list: the list of polynomials provided in a .ml file
      target_variable: variable to search for in polynomial terms

    Returns:
      the total number of terms containing the target variable across all polynomials
    """

    result = sum(get_no_terms_per_variable(input_polynomial=p,
                 target_variable=target_variable) for p in polynomial_list)

    return result


def total_degree_of_polynomial(input_polynomial: str) -> int:
    """the total degree of a polynomial is the maximum of the degrees
       of all terms in the polynomial, see: https://en.wikipedia.org/wiki/Degree_of_a_polynomial

    Args:
      input_polynomial: polynomial to process in .ml format

    Returns:
      the total degree of the given input polynomial
    """

    # get all terms of the input polynomial
    p_terms = input_polynomial.split('_')

    term_degree = []
    for term in p_terms:
        # extract all the degrees and add
        power_x1 = get_variable_degree(input_polynomial=term, target_variable='x1')
        power_x2 = get_variable_degree(input_polynomial=term, target_variable='x2')
        power_x3 = get_variable_degree(input_polynomial=term, target_variable='x3')

        term_sum = sum(power_x1) + sum(power_x2) + sum(power_x3)
        term_degree.append(term_sum)

    return max(term_degree)


def max_total_degree_of_polynomials(polynomial_list: List) -> int:
    """Get the maximum total degree across all polynomials
       in an input .ml file.

    Args:
      polynomial_list: the list of polynomials provided in a .ml file

    Returns:
      the maximum total degree of polynomials across all polynomials present
      in the file.
    """

    result = max(total_degree_of_polynomial(input_polynomial=p) for p in polynomial_list)

    return result


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
        # l_file = f"comp_times{f_number[0]}.txt"
        l_file = f"comp_times{f_number[0]}-perm{f_number[1]}.txt"

        if path_exists(input_path=os.path.join(path_input_data, file_name)) and path_exists(input_path=os.path.join(path_label_data, l_file)):
            polynomial_list = get_polynomials(
                input_file=os.path.join(path_input_data, file_name))[2:]

            nr_polynomials = len(polynomial_list)

            poly_sequence_label = get_label(input_file=os.path.join(path_label_data, l_file))

            # 3. extracting the degrees for each variable in the polynomials list from (1)
            processed_polynomials = [process_polynomial(
                input_polynomial=polynomial) for polynomial in polynomial_list]

            # print(f"{processed_polynomials}")

            # 4. getting the degrees for variables
            x1_deg = []
            x2_deg = []
            x3_deg = []

            for p in processed_polynomials:
                x1_deg.extend(p['x1'])
                x2_deg.extend(p['x2'])
                x3_deg.extend(p['x3'])

            # print(f"X1 degrees: {x1_deg}\nX2 degrees: {x2_deg}\nX3 degrees: {x3_deg}")

            # 5. get proportion of a variable in polynomials
            prop_x1 = get_proportions(processed_pol=processed_polynomials, target_variable='x1')
            prop_x2 = get_proportions(processed_pol=processed_polynomials, target_variable='x2')
            prop_x3 = get_proportions(processed_pol=processed_polynomials, target_variable='x3')
            # print(
            #     f"proportion of a variable in polynomials:\nX1: {prop_x1}\nX2: {prop_x2}\nX3: {prop_x3}")

            # 6. getting the proportion of a variable occurring in monomials
            total_no_terms_in_polynomials = get_number_of_terms(polynomial_list=polynomial_list)
            # print(f"Total no. terms: {total_no_terms_in_polynomials}")

            prop_mon_x1 = get_total_no_terms_per_variable(
                polynomial_list=polynomial_list, target_variable='x1')/total_no_terms_in_polynomials
            prop_mon_x2 = get_total_no_terms_per_variable(
                polynomial_list=polynomial_list, target_variable='x2')/total_no_terms_in_polynomials
            prop_mon_x3 = get_total_no_terms_per_variable(
                polynomial_list=polynomial_list, target_variable='x3')/total_no_terms_in_polynomials

            # print(
            #     f"proportion of a variable occurring in monomials:\nX1: {prop_mon_x1}\nX2: {prop_mon_x2}\nX3: {prop_mon_x3}")

            # 7. maximum total degree of polynomials
            max_total_degree = max_total_degree_of_polynomials(polynomial_list=polynomial_list)
            # print(f"Max total degree: {max_total_degree}")

            result_data.append({
                'file_id': f_number[0],
                'input_file': file_name,
                'label_file': l_file,
                'nr_polynomials': nr_polynomials,
                'max_total_degree': max_total_degree,
                'max_x1': max(x1_deg),
                'max_x2': max(x2_deg),
                'max_x3': max(x3_deg),
                'prop_x1': prop_x1,
                'prop_x2': prop_x2,
                'prop_x3': prop_x3,
                'prop_mon_x1': prop_mon_x1,
                'prop_mon_x2': prop_mon_x2,
                'prop_mon_x3': prop_mon_x3,
                'label': poly_sequence_label})

        if i % 1000 == 0:
            print(f"-- Processed {i} files!")

    print(f"- Output contains: {len(result_data)} processed polynomials.")

    # save results to .csv file
    df_output_dataset = pd.DataFrame(result_data)
    df_output_dataset.to_csv(path_output_file, sep='\t')

    print(f"- Output file saved at: {path_output_file}")
    print(f"- Execution completed in {time.time() - start_time} seconds.")


def main_test():
    target_polynomial = "3\n(abstract ML syntax version of) hong_1.rlqe.redlog\nx1^2 _ x2^2 _ x3^2 _ $\nx1*x2*x3 _ $\nx1^5"
    print(f"Target polynomial:\n{target_polynomial}")

    # polynomial_list = get_polynomials(input_file=os.path.join(INPUT_DIR, file_path))[2:]

    # 1. get the individual polynomials
    # skip the first two lines as they give the number variables of polynomials
    # and the MetiTarski file name
    polynomial_list = get_polynomials_test(input_text=target_polynomial)[2:]
    print(f"Polynomials on file:\n{polynomial_list}")

    # 2. get the number of polynomials
    no_polynomials = len(polynomial_list)
    print(f"There are {no_polynomials} polynomials on file!")

    # 3. extracting the degrees for each variable in the polynomials list from (1)
    processed_polynomials = [process_polynomial(
        input_polynomial=polynomial) for polynomial in polynomial_list]

    print(f"{processed_polynomials}")

    # 4. getting the degrees for variables
    x1_deg = []
    x2_deg = []
    x3_deg = []

    for p in processed_polynomials:
        x1_deg.extend(p['x1'])
        x2_deg.extend(p['x2'])
        x3_deg.extend(p['x3'])

    print(f"X1 degrees: {x1_deg}\nX2 degrees: {x2_deg}\nX3 degrees: {x3_deg}")

    # 5. get proportion of a variable in polynomials
    prop_x1 = get_proportions(processed_pol=processed_polynomials, target_variable='x1')
    prop_x2 = get_proportions(processed_pol=processed_polynomials, target_variable='x2')
    prop_x3 = get_proportions(processed_pol=processed_polynomials, target_variable='x3')
    print(f"proportion of a variable in polynomials:\nX1: {prop_x1}\nX2: {prop_x2}\nX3: {prop_x3}")

    # 6. getting the proportion of a variable occurring in monomials
    total_no_terms_in_polynomials = get_number_of_terms(polynomial_list=polynomial_list)
    print(f"Total no. terms: {total_no_terms_in_polynomials}")

    prop_mon_x1 = get_total_no_terms_per_variable(
        polynomial_list=polynomial_list, target_variable='x1')/total_no_terms_in_polynomials
    prop_mon_x2 = get_total_no_terms_per_variable(
        polynomial_list=polynomial_list, target_variable='x2')/total_no_terms_in_polynomials
    prop_mon_x3 = get_total_no_terms_per_variable(
        polynomial_list=polynomial_list, target_variable='x3')/total_no_terms_in_polynomials

    print(
        f"proportion of a variable occurring in monomials:\nX1: {prop_mon_x1}\nX2: {prop_mon_x2}\nX3: {prop_mon_x3}")

    # 7. maximum total degree of polynomials
    for polynomial in polynomial_list:
        d_polynomial = total_degree_of_polynomial(input_polynomial=polynomial)
        print(f"{polynomial}\ttotal degree: {d_polynomial}")

    max_total_degree = max_total_degree_of_polynomials(polynomial_list=polynomial_list)
    print(f"Max total degree: {max_total_degree}")


if __name__ == '__main__':
    main()
