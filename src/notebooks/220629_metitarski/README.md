# MetiTarski Problem using ML

## Data

* input: https://github.com/SRI-CSL/CoProver/tree/main/data/polynomials/polys
* label: https://github.com/SRI-CSL/CoProver/tree/main/data/polynomials/comp-times

## Papers

### Efficient Projection Orders for CAD

**Authors**
* Andreas Dolzmann
* Andreas Seidl
* Thomas Sturm

### Using Machine Learning to Improve Cylindrical Algebraic Decomposition

**Authors**
* Zongyan Huang
* Matthew England
* David J. Wilson
* James Bridge
* James H. Davenport
* Lawrence C. Paulson

## Additional Links

* https://f-charton.github.io/polynomial-roots/
* Linear algebra with transformers, see https://arxiv.org/pdf/2112.01898.pdf
* https://medium.com/analytics-vidhya/solving-differential-equations-with-transformers-21648d3a1695

## Sample Output

```json
[
    {
        "file_id": 3940,
        "input_file": "poly3940.txt.ml",
        "label": 0,
        "label_file": "comp_times3940.txt",
        "max_x1": 1,
        "max_x2": 1,
        "max_x3": 1,
        "nr_polynomials": 4,
        "prop_mon_x1": 0.4,
        "prop_mon_x2": 0.4,
        "prop_mon_x3": 0.2,
        "prop_x1": 0.5,
        "prop_x2": 0.5,
        "prop_x3": 0.25
    },
    {
        "file_id": 5554,
        "input_file": "poly5554.txt.ml",
        "label": 4,
        "label_file": "comp_times5554.txt",
        "max_x1": 10,
        "max_x2": 9,
        "max_x3": 1,
        "nr_polynomials": 12,
        "prop_mon_x1": 0.4186046511627907,
        "prop_mon_x2": 0.5116279069767442,
        "prop_mon_x3": 0.06976744186046512,
        "prop_x1": 0.6666666666666666,
        "prop_x2": 0.6666666666666666,
        "prop_x3": 0.25
    },
    {
        "file_id": 4063,
        "input_file": "poly4063.txt.ml",
        "label": 5,
        "label_file": "comp_times4063.txt",
        "max_x1": 1,
        "max_x2": 1,
        "max_x3": 1,
        "nr_polynomials": 9,
        "prop_mon_x1": 0.3076923076923077,
        "prop_mon_x2": 0.3076923076923077,
        "prop_mon_x3": 0.38461538461538464,
        "prop_x1": 0.4444444444444444,
        "prop_x2": 0.4444444444444444,
        "prop_x3": 0.5555555555555556
    },
    {
        "file_id": 4732,
        "input_file": "poly4732.txt.ml",
        "label": 2,
        "label_file": "comp_times4732.txt",
        "max_x1": 4,
        "max_x2": 2,
        "max_x3": 1,
        "nr_polynomials": 7,
        "prop_mon_x1": 0.5,
        "prop_mon_x2": 0.2,
        "prop_mon_x3": 0.3,
        "prop_x1": 0.42857142857142855,
        "prop_x2": 0.2857142857142857,
        "prop_x3": 0.42857142857142855
    },
    {
        "file_id": 5205,
        "input_file": "poly5205.txt.ml",
        "label": 5,
        "label_file": "comp_times5205.txt",
        "max_x1": 12,
        "max_x2": 6,
        "max_x3": 1,
        "nr_polynomials": 6,
        "prop_mon_x1": 0.44,
        "prop_mon_x2": 0.44,
        "prop_mon_x3": 0.12,
        "prop_x1": 0.5,
        "prop_x2": 0.3333333333333333,
        "prop_x3": 0.5
    }
]
```

## Sample run

```python
$ python create_dataset_20220805.py
```

```bash
- Input data is located at: /PATH_TO_COPROVER_GIT_REPO/CoProver/data/polynomials/balanced/polys
- Respective label information is located at: /PATH_TO_COPROVER_GIT_REPO/CoProver/data/polynomials/balanced/comp-times
- There are 41370-input files ending with .ml.
Processed: 100 files.
Processed: 200 files.
Processed: 300 files.
Processed: 400 files.
Processed: 500 files.
Processed: 600 files.
Processed: 700 files.
Processed: 800 files.
Processed: 900 files.
Processed: 1000 files.
.
.
.
- Execution duration: 0:00:37.364905
(signal-env) CSL-CAS15861:220629_metitarski e32704$
```

## Dataset Preprocessing for Transformer Architectures

```python
python preprocess_data.py --data_dir /PATH_TO_COPROVER_GIT_REPO/CoProver/data/polynomials/polys/ --label_dir /PATH_TO_COPROVER_GIT_REPO/CoProver/data/polynomials/comp-times/ --output_file /PATH_TO_COPROVER_GIT_REPO/CoProver/data/metitarski/metitarski_dataset_transformers_v1.csv
```

```bash
- Input: /CoProver/data/polynomials/polys/
- Labels: /CoProver/data/polynomials/comp-times/
- There are 7200-input files ending with .ml.
-- Processed 0 files!
-- Processed 1000 files!
-- Processed 2000 files!
-- Processed 3000 files!
-- Processed 4000 files!
-- Processed 5000 files!
-- Processed 6000 files!
-- Processed 7000 files!
- Output contains: 6895 processed polynomials.
- Output file saved at: /CoProver/data/metitarski/metitarski_dataset_transformers_v1.csv
- Execution completed in 4.676846742630005 seconds.
```
