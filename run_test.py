#!/usr/bin/env python3

import os
import sys
import time

def get_time(start_time):
    return int((time.time() - start_time) * 1000)

def compare_plates(p1, p2):
    if (len(p1) != 7) or (len(p2) != 7):
        return 0

    score = 0
    for i in range(7):
        if (i != 4) and (p1[i] == p2[i]):
            score += 1;
    return score

start_time = time.time();

files = os.listdir("./test_images")

correct = 0
wrong = 0
multiple = 0
sum = 0

for f in files:
    reads = [r for r in os.popen("./read_plate.py -silent -no-image -input=test_images/" + f).read().strip().split("\n") if len(r) == 7]
    plate = f.split("_")[0]

    best = "###-###"
    best_score = 0

    for r in reads:
        curr = compare_plates(plate, best)
        new = compare_plates(plate, r)

        if (curr < new):
            best = r
            best_score = new

    print(plate, "---", best, "Score:", best_score, "Reads:", len(reads));

    if (len(reads) == 1) and (best_score == 6):
        correct += 1
    elif (best_score == 6):
        multiple += 1
    else:
        wrong += 1
    sum += best_score

print("Test done in", get_time(start_time) / 1000, "s")
print("Total:", len(files))
print("Correct:", correct)
print("Multiple:", multiple)
print("Wrong:", wrong)
print("Average score:", sum / len(files))
