"""CoProver: The Cueology of Proof

@description: Data processing for MetiTarski variable ordering problem

@copyright: SRI International, 2022
"""

import os
import sys

import re
import numpy as np
import pandas as pd
import json  # only for printing the dictionary, to be removed

from os import listdir
from os.path import isfile, join
from typing import List, Any, Dict
from datetime import datetime

# /CoProver
ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir))
# /CoProver/data
DATA_DIR = os.path.join(ROOT_DIR, 'data')
# /CoProver/data/polynomials
# V2 working on the balanced data found under the /CoProver/data/polynomials/balanced
POLYNOMIALS_DIR = os.path.join(DATA_DIR, 'polynomials', 'balanced')
# /CoProver/data/polynomials/polys
INPUT_DIR = os.path.join(POLYNOMIALS_DIR, 'polys')
# /CoProver/data/polynomials/comp-times
LABEL_DIR = os.path.join(POLYNOMIALS_DIR, 'comp-times')
# Output CSV file, i.e., DataFrame
OUTPUT_CSV = os.path.join(DATA_DIR, 'metitarski', 'metitarski_dataset_v2.csv')

# FEATURE LIST
# Number of polynomials.
# Maximum total degree of polynomials.
# Maximum degree of x0 among all polynomials.
# Maximum degree of x1 among all polynomials.
# Maximum degree of x2 among all polynomials.
# Proportion of x0 occuring in polynomials.
# Proportion of x1 occuring in polynomials.
# Proportion of x2 occuring in polynomials.
# Proportion of x0 occuring in monomials.
# Proportion of x1 occuring in monomials.
# Proportion of x2 occuring in monomials.


def path_exists(input_path: str) -> bool:
    return os.path.exists(input_path)


def get_polynomials(input_file: str) -> List:
    res = []

    with open(input_file) as f:
        res = [f_line.rstrip() for f_line in f]

    return res


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


def get_files_paths(input_dir: str, ends_with: str) -> List:

    file_paths = [f for f in listdir(input_dir) if isfile(
        join(input_dir, f)) and f.endswith(ends_with)]

    return file_paths


def get_proportions(processed_pol: Dict, target_variable: str) -> float:
    cnt_target_occurrences = 0

    for p in processed_pol:
        # check whether target variable existed in p-polynomial
        s_v = sum(1 for i in p[target_variable] if i != 0)
        if s_v != 0:
            cnt_target_occurrences += 1

    # print(target_variable, cnt_target_occurrences, len(processed_pol))
    return cnt_target_occurrences/len(processed_pol) if len(processed_pol) > 0 else 0


def get_number_of_monomials(processed_pol: Dict) -> int:

    cnt_x1 = 0
    cnt_x2 = 0
    cnt_x3 = 0

    for p in processed_pol:
        cnt_x1 += sum(1 for i in p['x1'] if i != 0)
        cnt_x2 += sum(1 for i in p['x2'] if i != 0)
        cnt_x3 += sum(1 for i in p['x3'] if i != 0)

    return cnt_x1, cnt_x2, cnt_x3


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
    start_time = datetime.now()
    print(
        f"- Input data is located at: {INPUT_DIR}\n- Respective label information is located at: {LABEL_DIR}")

    input_file_paths = get_files_paths(input_dir=INPUT_DIR, ends_with=".ml")
    print(f"- There are {len(input_file_paths)}-input files ending with .ml.")

    # p_2 = process_polynomial(input_polynomial=SAMPLE_INPUT_2)
    # p_3 = process_polynomial(input_polynomial=SAMPLE_INPUT_3)
    # print(p_2)
    # print(p_3)
    merge_data = []

    cnt = 0
    for file_path in input_file_paths:
        f_number = get_numbers_from_str(file_path)
        l_file = f"comp_times{f_number[0]}-perm{f_number[1]}.txt"
        # print(file_path)
        # print(f_number)
        # print(l_file)
        #
        # cnt += 1
        # if cnt == 3:
        #     break
        # process data only if both files exist
        # at present, there are missing files in the comp-times folder
        if path_exists(input_path=os.path.join(INPUT_DIR, file_path)) and path_exists(input_path=os.path.join(LABEL_DIR, l_file)):
            # 1. extracting all polynomials from a given input file
            polynomial_list = get_polynomials(input_file=os.path.join(INPUT_DIR, file_path))[2:]

            nr_polynomials = len(polynomial_list)
            # print(polynomial_list)
            # print(f_number, l_file)
            # print("---")

            # 2. extracting the degrees for each variable in the polynomials list from (1)
            processed_polynomials = [process_polynomial(
                input_polynomial=polynomial) for polynomial in polynomial_list]

            # print(len(processed_polynomials), nr_polynomials)
            # 3. getting the degrees for variables
            x1_deg = []
            x2_deg = []
            x3_deg = []

            for p in processed_polynomials:
                x1_deg.extend(p['x1'])
                x2_deg.extend(p['x2'])
                x3_deg.extend(p['x3'])

            # print(f"x1: {x1_deg}")
            # print(f"x2: {x2_deg}")
            # print(f"x3: {x3_deg}")
            tot_x1, tot_x2, tot_x3 = get_number_of_monomials(processed_pol=processed_polynomials)
            no_monomials = tot_x1 + tot_x2 + tot_x3

            # adding the resulting dictionary to the list
            merge_data.append({
                'file_id': f_number[0],
                'input_file': file_path,
                'label_file': l_file,
                'nr_polynomials': nr_polynomials,
                'max_x1': max(x1_deg),
                'max_x2': max(x2_deg),
                'max_x3': max(x3_deg),
                'prop_x1': get_proportions(processed_pol=processed_polynomials, target_variable='x1'),
                'prop_x2': get_proportions(processed_pol=processed_polynomials, target_variable='x2'),
                'prop_x3': get_proportions(processed_pol=processed_polynomials, target_variable='x3'),
                'prop_mon_x1': tot_x1/no_monomials if no_monomials > 0 else 0,
                'prop_mon_x2': tot_x2/no_monomials if no_monomials > 0 else 0,
                'prop_mon_x3': tot_x3/no_monomials if no_monomials > 0 else 0,
                'label': get_label(input_file=os.path.join(LABEL_DIR, l_file))})

            cnt += 1
            # if cnt == 5:
            #     break

            if cnt % 100 == 0:
                print(f"Processed: {cnt} files.")

    # print(merge_data)
    # print(json.dumps(merge_data, sort_keys=True, indent=4))

    # create a dataframe and save it
    df_output_dataset = pd.DataFrame(merge_data)
    df_output_dataset.to_csv(OUTPUT_CSV, sep='\t')

    end_time = datetime.now()
    print(f"- Execution duration: {end_time - start_time}")


def test_res():
    input_str = "brilandbrilandbrilandbrilandb"
    occurrences_v = [v.start() for v in re.finditer('b', input_str)]
    print(input_str)
    print(occurrences_v)
    # a = [1, 5, 8, 9]
    #
    for i in range(len(occurrences_v)):
        if i == len(occurrences_v) - 1:
            print(occurrences_v[i])
            print(input_str[occurrences_v[i]:])
        else:
            print(occurrences_v[i], occurrences_v[i+1])
            print(input_str[occurrences_v[i]:occurrences_v[i+1]])


if __name__ == '__main__':
    main()
