import gzip

"""
Use this to convert the GZ feature files into text forms suitable for training a vocabulary
"""


for prefix in ["cmdpred_N3.prelude", "cmdpred_N3.pvslib"]:
    with gzip.open("{}.tsv.gz".format(prefix), 'rt') as f:
        with open("{}.txt".format(prefix), 'w') as out_f:
            for line in f:
                line = line.replace(",", " ")            
                for datum in line.split("\t"):
                    if "/" in datum and "#" in datum:
                        continue # Avoid URIs
                    out_f.write(datum.strip())
                    out_f.write(" ")
                    out_f.write("\n")
