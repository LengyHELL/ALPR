#!/usr/bin/env python3

import sys
import numpy as np
import cv2

input_name, input_ext = sys.argv[1].split(".")

blur_buffer_size = int(sys.argv[2])

I = cv2.imread(input_name + "." + input_ext)

if (blur_buffer_size >= 3):
    G = cv2.GaussianBlur(I, (blur_buffer_size, blur_buffer_size), 0)
else:
    G = I

G = cv2.cvtColor(G, cv2.COLOR_BGR2GRAY)

thr, bin = cv2.threshold(G, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

O = cv2.Canny(G, thr, 0.5 * thr)

cnt, hier = cv2.findContours(O, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

ch = zip(cnt, hier[0])

ch = sorted(ch, key=lambda x : cv2.contourArea(x[0]), reverse=True)

ch = [c for c in ch if (c[1][0] == -1) and (c[1][1] == -1)][:15]

S = I.copy()

plates = []
for c, h in ch:
    epsilon = 0.02 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)
    if (len(approx) == 4):
        plates.append(c)
        #print(h)
    else:
        S = cv2.drawContours(S, [c], -1, (0, 0, 255), 1)

m = np.zeros(G.shape, np.uint8)
masked = cv2.drawContours(m, plates, 0, 255, -1,)
masked = cv2.bitwise_and(O, O, mask=m)

D = I.copy()

index = 0
for p in plates:
    x, y, w, h = cv2.boundingRect(p)

    crop = masked[y:y+h, x:x+w]
    real = D[y:y+h, x:x+w]

    cnt2, hier2 = cv2.findContours(crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if hier2 is None:
        continue

    ch2 = zip(cnt2, hier2[0])
    ch2 = sorted(ch2, key=lambda x : cv2.contourArea(x[0]), reverse=True)
    ch2 = [c for c in ch2 if c[1][1] != -1]

    #for c in ch2:
    #    print(c[1])

    cnt2 = [c[0] for c in ch2][:6]

    #print(len(cnt2))

    if len(cnt2) >= 6:
        D = cv2.drawContours(D, [p], -1, (0, 255, 0), 2)
        real = cv2.drawContours(real, cnt2, -1, (255, 0, 0), 1)

    #cv2.imwrite(input_name + "_plate" + str(index) + "." + input_ext, real)
    index += 1

#cv2.imwrite(input_name + "_processed." + input_ext, O)
#cv2.imwrite(input_name + "_detected." + input_ext, D)

cv2.imshow("Processed image", O.astype(np.uint8))
cv2.imshow("Skipped candidates", S.astype(np.uint8))
cv2.imshow("Detected plates", D.astype(np.uint8))

cv2.waitKey()
cv2.destroyAllWindows()
