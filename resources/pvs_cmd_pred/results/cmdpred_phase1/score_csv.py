import pandas as pd
import sys

csv_file = sys.argv[1]
print(csv_file)

num_correct = 0
total = 0
header_read = False
with open(csv_file, 'r') as f:
    for line in f:
        if header_read is False:
            header_read = True
        else:
            (guess, gold) = line.split(",")[0:2]
            guess, gold = guess.strip(), gold.strip()
            print(guess, gold, guess==gold)
            if guess == gold:
                num_correct += 1
            total += 1
print(num_correct, total)
print(num_correct / total)
