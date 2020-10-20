#!/usr/bin/env python3

import sys
import numpy as np
import cv2

def is_inside(inside, outside):
    in_point = 0;
    for i in inside:
        is_in = cv2.pointPolygonTest(outside, tuple(i[0]), False)
        if is_in > 0:
            return True
    return False

img = cv2.imread(sys.argv[1])

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

thr = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0]

img_edges = cv2.Canny(img_gray, thr, 0.5 * thr)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
img_edges = cv2.dilate(img_edges, kernel)

cnt, hier = cv2.findContours(img_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

ch = [[cnt[i], hier[0][i]] for i in range(len(cnt))]

good = []
for i in range(len(ch)):
    ok = True
    for j in range(len(ch)):
        if (i != j) and is_inside(ch[i][0], ch[j][0]):
            ok = False
            break
    if ok:
        good.append(ch[i])

cnt = [g[0] for g in good]

idx = -1
for c in cnt:
    idx += 1
    x, y, w, h = cv2.boundingRect(c)
    cv2.imwrite(str(idx) + ".jpg", img[y:y+h, x:x+w].astype(np.uint8))
    img = cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0, 0), 1)

#cv2.imshow("Processed", img_edges.astype(np.uint8))
#cv2.imshow("Characters", img.astype(np.uint8))
#cv2.imshow("cpy", cpy.astype(np.uint8))

cv2.waitKey()
cv2.destroyAllWindows()
