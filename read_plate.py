#!/usr/bin/env python3

import sys
import numpy as np
import cv2
import time

def get_time(start_time):
    return int((time.time() - start_time) * 1000)

def is_inside(inside, outside, limit_val=-1):
    point_limit = limit_val * len(inside)
    if limit_val < 0:
        point_limit = 1
    in_point = 0;
    for i in inside:
        is_in = cv2.pointPolygonTest(outside, tuple(i[0]), False)
        if is_in >= 0:
            in_point += 1
            if in_point >= point_limit:
                return True
    return False

start_time = time.time()

arg = {}
for a in sys.argv[1:]:
    if (a[0] == "-"):
        a = a[1:]
        a = a.split("=")
        if len(a) == 2:
            arg[a[0]] = a[1]
        elif len(a) == 1:
            arg[a[0]] = ""
        else:
            sys.exit(3)
    else:
        sys.exit(2)

if "input" not in arg:
    sys.exit(1)

input_name, input_ext = arg["input"].split(".")

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

blur_limit = float(15)
if "blur-limit" in arg:
    blur_limit = float(arg["blur-limit"])

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

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
img_edges = cv2.morphologyEx(img_edges, cv2.MORPH_CLOSE, kernel)

cnt, hier = cv2.findContours(img_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

ch = []
if False:
    ch = [[cv2.convexHull(cnt[i], False), i] for i in range(len(cnt))]
else:
    ch = [[cnt[i], i] for i in range(len(cnt))]

ch_top = sorted(ch, key=lambda x : cv2.contourArea(x[0]), reverse=True)[:50]
print("Found", len(ch), "contours.", get_time(start_time), "ms")

img_filtered = img.copy()

possible = []
for t in ch_top:
    inner = 0
    for b in ch:
        if is_inside(b[0], t[0]):
            inner += 1
        if inner >= 6:
            possible.append(t[1])
            break
    if inner < 6:
        img_filtered = cv2.drawContours(img_filtered, [t[0]], -1, (0, 126, 255), 1)

ch = [ch[p] for p in possible]

#ch = [[cv2.convexHull(c[0]), c[1]] for c in ch]

plates = []
for c, idx in ch:
    og = c
    rect = cv2.minAreaRect(c)

    box = cv2.boxPoints(rect)
    c = np.int0(box)

    if ((cv2.contourArea(c) / cv2.contourArea(og)) - 1) <= 0.2:
        desired = 520 / 110;
        current = max(rect[1]) / min(rect[1])
        margin = 0.3

        if desired * (1 - margin) <= current <= desired * (1 + margin):
            plates.append([c, og])
        else:
            img_filtered = cv2.drawContours(img_filtered, [c], -1, (0, 0, 255), 1)
    else:
        img_filtered = cv2.drawContours(img_filtered, [c], -1, (0, 0, 255), 1)

good = []
for i in range(len(plates)):
    ok = True
    for j in range(len(plates)):
        if (i != j) and is_inside(plates[j][1], plates[i][1], 1):
            ok = False
            break
    if ok:
        good.append(plates[i])
    else:
        img_filtered = cv2.drawContours(img_filtered, [plates[i][1]], -1, (255, 255, 0), 1)

plates = good

img_detected = img.copy()
candidates = []

index = 0
for p, og in plates:
    mask = np.zeros(img_gray.shape, np.uint8)
    img_masked = cv2.drawContours(mask, [p], 0, 255, -1,)
    img_masked = cv2.bitwise_and(img_edges, img_edges, mask=mask)

    cv2.drawContours(img_masked, [og], 0, 0, 2)

    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    #print(kernel)
    #img_masked = cv2.dilate(img_masked, kernel)

    x, y, w, h = cv2.boundingRect(p)
    crop_masked = img_masked[y:y+h, x:x+w]
    crop_detected = img_detected[y:y+h, x:x+w]

    cnt, hier = cv2.findContours(crop_masked, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)

    if hier is None:
        img_filtered = cv2.drawContours(img_filtered, [p], -1, (255, 0, 255), 1)
        continue

    hier = hier[0]
    ch = [[cnt[i], hier[i]] for i in range(len(cnt)) if (hier[i][0] != -1) or (hier[i][1] != -1)]

    for i in range(len(ch)):
        ch[i][0] = cv2.convexHull(ch[i][0], False)

    good = []
    for i in range(len(ch)):
        ok = True
        for j in range(len(ch)):
            if (i != j) and is_inside(ch[i][0], ch[j][0], 0.8):
                ok = False
                break
        if ok:
            good.append(ch[i])

    ch = sorted(good, key=lambda x : cv2.contourArea(x[0]) * cv2.boundingRect(x[0])[3], reverse=True)[:6]

    if (len(ch) >= 6):
        chars = []
        img_detected = cv2.drawContours(img_detected, [og], -1, (0, 255, 0), 2)
        cnt = [c[0] for c in ch]
        #crop_detected = cv2.drawContours(crop_detected, cnt, -1, (255, 0, 0), 1)
        num = -1
        for c in cnt:
            num += 1
            #box = cv2.boxPoints(cv2.minAreaRect(c))
            #box = np.int0(box)
            #crop_detected = cv2.drawContours(crop_detected, [box], -1, (255, 0, 0), 1)
            x, y, w, h = cv2.boundingRect(c)
            crop_detected = cv2.rectangle(crop_detected, (x,y), (x+w,y+h), (255, 0, 0), 1)
            chars.append([crop_detected.copy()[y:y+h, x:x+w], x])
        chars = sorted(chars, key=lambda x : x[1])
        candidates.append([c[0] for c in chars])
        index += 1
        #cv2.imshow("Last plate", crop_masked.astype(np.uint8))
    else:
        img_filtered = cv2.drawContours(img_filtered, [p], -1, (0, 255, 255), 1)

print(index, "plates found.")

idx = 0
t_num = "0123456789"
t_char = "abcdefghijklmnoprstuvwxyz"
for cnd in candidates:
    idx += 1
    plate = ""
    pos = 0
    for c in cnd:
        if pos > 2:
            templates = t_num
        else:
            templates = t_char
        pos += 1

        vals = []
        for t in templates:
            template = cv2.imread("templates/" + t + ".jpg")
            h, w, col = c.shape
            template = cv2.resize(template, (w, h), interpolation=cv2.INTER_AREA)

            t_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            c_gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)

            t_gray = cv2.adaptiveThreshold(t_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 0)
            c_gray = cv2.adaptiveThreshold(c_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 0)
            #template = cv2.threshold(template, 126, 255, cv2.THRESH_BINARY)[1]

            #cv2.imshow("org", c_gray.astype(np.uint8))
            #cv2.imshow("tmp", t_gray.astype(np.uint8))

            vals.append([t, cv2.matchTemplate(t_gray, c_gray, cv2.TM_SQDIFF)[0][0]])
        plate += sorted(vals, key=lambda x : x[1])[0][0]
    plate = plate.upper()
    plate = plate[:3] + "-" + plate[3:]
    print("Plate " + str(idx) + " number:", plate)

print("Executed in %.0f ms" % ((time.time() - start_time) * 1000))

if "no-image" not in arg:
    cv2.imshow("Processed image", img_edges.astype(np.uint8))
    cv2.imshow("Filtered candidates", img_filtered.astype(np.uint8))

    if (index > 0):
        cv2.imshow("Detected plates", img_detected.astype(np.uint8))
        #cv2.imshow("First detected plate", crop_masked.astype(np.uint8))

    cv2.waitKey()
    cv2.destroyAllWindows()
