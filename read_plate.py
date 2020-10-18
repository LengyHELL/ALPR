#!/usr/bin/env python3

import sys
import numpy as np
import cv2

if (len(sys.argv) <= 1):
    sys.exit(1)

input_name, input_ext = sys.argv[1].split(".")

def is_inside(inside, outside):
    in_point = 0;
    for i in inside:
        is_in = cv2.pointPolygonTest(outside, tuple(i[0]), False)
        if is_in > 0:
            return True
    return False


img = cv2.imread(input_name + "." + input_ext)

if img is None:
    sys.exit(1)

h, w, c = img.shape

max_width = 1920
max_height = 1080
if (w > max_width) or (h > max_height):
    ratio = min(max_width / w, max_height / h)
    new_size = (round(w * ratio), round(h * ratio))
    print("Resizing image, new size: %dx%d, %.2f%%"%(new_size[0], new_size[1], ratio))
    img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

img_blur = cv2.fastNlMeansDenoisingColored(img, None, 15, 10, 7, 21)

blur_limit = float(100)
if len(sys.argv) >= 3:
    blur_limit = float(sys.argv[2])

ok = False
while ok == False:
    img_blur = cv2.GaussianBlur(img_blur, (3, 3), 0)
    detected_blur = cv2.Laplacian(img_blur, cv2.CV_64F).var() * 100000 / (img.shape[0] * img.shape[1])
    print("Blur value: %.2f"%(detected_blur))
    if detected_blur <= blur_limit:
        ok = True

img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)

thr = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0]

img_edges = cv2.Canny(img_gray, thr, 0.5 * thr)

cnt, hier = cv2.findContours(img_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
hier = hier[0]

ch = [[cnt[i], hier[i], i] for i in range(len(cnt))]
ch_top = sorted(ch, key=lambda x : cv2.contourArea(x[0]), reverse=True)[:40]

img_filtered = img.copy()

possible = []
for i in range(len(ch_top)):
    inner = 0
    for c in ch:
        if c[1][3] == ch_top[i][2]:
            inner += 1
        if inner >= 6:
            possible.append(i)
            break
    if inner < 6:
        img_filtered = cv2.drawContours(img_filtered, [ch_top[i][0]], -1, (0, 126, 255), 1)

ch = [ch_top[p] for p in possible]

good = []
for i in range(len(ch)):
    ok = True
    for j in range(len(ch)):
        if (i != j) and is_inside(ch[j][0], ch[i][0]):
            ok = False
            break
    if ok:
        good.append(ch[i])
    else:
        img_filtered = cv2.drawContours(img_filtered, [ch[i][0]], -1, (255, 255, 0), 1)

ch = good

plates = []
for c, h, idx in ch:
    found = False
    arc_length = 0.01 * cv2.arcLength(c, True)
    c = cv2.convexHull(c, False)

    for i in range(0, 21):
        epsilon = i * arc_length
        approx = cv2.approxPolyDP(c, epsilon, True)
        approx = [a[0] for a in approx]

        if len(approx) == 4:
            sides = []
            for a in range(len(approx)):
                b = a + 1
                if b >= len(approx):
                    b = 0
                t1 = approx[a][0] - approx[b][0]
                t2 = approx[a][1] - approx[b][1]
                sides.append((t1 * t1 + t2 * t2)**0.5)

            side_a = (sides[0] + sides[2]) / 2
            side_b = (sides[1] + sides[3]) / 2

            desired = 520 / 110;
            current = max(side_a, side_b) / min(side_a, side_b)
            margin = 0.3

            if desired * (1 - margin) <= current <= desired * (1 + margin):
                plates.append(c)
                found = True
                break

    if not found:
        img_filtered = cv2.drawContours(img_filtered, [c], -1, (0, 0, 255), 1)

img_detected = img.copy()

index = 0
for p in plates:
    mask = np.zeros(img_gray.shape, np.uint8)
    img_masked = cv2.drawContours(mask, [p], 0, 255, -1,)
    img_masked = cv2.bitwise_and(img_edges, img_edges, mask=mask)

    x, y, w, h = cv2.boundingRect(p)
    crop_masked = img_masked[y:y+h, x:x+w]
    crop_detected = img_detected[y:y+h, x:x+w]

    cnt, hier = cv2.findContours(crop_masked, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)

    if hier is None:
        img_filtered = cv2.drawContours(img_filtered, [p], -1, (255, 0, 255), 1.5)
        continue

    hier = hier[0]
    ch = [[cnt[i], hier[i]] for i in range(len(cnt)) if ((hier[i][0] != -1) or (hier[i][1] != -1)) and (hier[i][3] != -1)]

    for i in range(len(ch)):
        ch[i][0] = cv2.convexHull(ch[i][0], False)

    good = []
    for i in range(len(ch)):
        ok = True
        for j in range(len(ch)):
            if (i != j) and is_inside(ch[i][0], ch[j][0]):
                ok = False
                break
        if ok:
            good.append(ch[i])

    ch = sorted(good, key=lambda x : cv2.contourArea(x[0]), reverse=True)[:6]

    if len(ch) >= 6:
        img_detected = cv2.drawContours(img_detected, [p], -1, (0, 255, 0), 2)
        cnt = [c[0] for c in ch]
        #crop_detected = cv2.drawContours(crop_detected, cnt, -1, (255, 0, 0), 1)
        for c in cnt:
            #box = cv2.boxPoints(cv2.minAreaRect(c))
            #box = np.int0(box)
            #crop_detected = cv2.drawContours(crop_detected, [box], -1, (255, 0, 0), 1)
            x, y, w, h = cv2.boundingRect(c)
            crop_detected = cv2.rectangle(crop_detected, (x,y), (x+w,y+h), (255, 0, 0), 1)
    else:
        img_filtered = cv2.drawContours(img_filtered, [p], -1, (0, 255, 255), 1)

    index += 1

print(index, "plates found.")

cv2.imshow("Processed image", img_edges.astype(np.uint8))
cv2.imshow("Filtered candidates", img_filtered.astype(np.uint8))

if (len(plates) > 0):
    cv2.imshow("Detected plates", img_detected.astype(np.uint8))
    cv2.imshow("First detected plate", crop_masked.astype(np.uint8))

cv2.waitKey()
cv2.destroyAllWindows()
