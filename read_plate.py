#!/usr/bin/env python3

import sys
import numpy as np
import cv2

input_name, input_ext = sys.argv[1].split(".")

I = cv2.imread(input_name + "." + input_ext)

O = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

thr, bin = cv2.threshold(O, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

O = cv2.Canny(O, thr, 0.5 * thr)

cnt, hier = cv2.findContours(O, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cnt = sorted(cnt, key=cv2.contourArea, reverse=True)[:10]

plates = []
for c in cnt:
    epsilon = 0.02 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)
    if len(approx) == 4:
        plates.append(c)

index = 0
for p in plates:
    x, y, w, h = cv2.boundingRect(p)
    crop = I[y:y+h, x:x+w]
    cv2.imwrite(input_name + "_plate" + str(index) + "." + input_ext, crop)
    index += 1

D = cv2.drawContours(I, plates, -1, (0, 255, 0), 2)

cv2.imwrite(input_name + "_processed." + input_ext, O)
cv2.imwrite(input_name + "_detected." + input_ext, D)
