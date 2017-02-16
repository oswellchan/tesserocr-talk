from PIL import Image
from tesserocr import PyTessBaseAPI, RIL, PSM, Orientation
import time
import numpy as np
import cv2
import math
import scipy.ndimage
import matplotlib.pyplot as plt
import threading
from multiprocessing import Process, Queue

THRESHOLD_WHITE = 200
WEIGHT = 5


def measure_timing(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print('time: ', time.time() - start)
        return result
    return wrapper


@measure_timing
def draw_boxes(boxes, image, file_name='output.jpg', color=(255, 0, 0)):
    img = image.copy()
    pix = img.load()
    width, height = img.size
    for (_, box, _, _) in boxes:
        x = int(box['x'])
        y = int(box['y'])
        w = int(box['w'])
        h = int(box['h'])

        for i in range(w):
            pix[x + i, y] = color
            pix[x + i, y + h - 1] = color

        for i in range(h):
            pix[x, y + i] = color
            pix[x + w - 1, y + i] = color

    img.save(file_name)


def rotate_to_upright(image, api):
    api.SetPageSegMode(PSM.OSD_ONLY)
    api.SetImage(image)

    os = api.DetectOS()
    if os:
        if os['orientation'] == Orientation.PAGE_RIGHT:
            image = image.rotate(90, expand=True)

        if os['orientation'] == Orientation.PAGE_LEFT:
            image = image.rotate(270, expand=True)

        if os['orientation'] == Orientation.PAGE_DOWN:
            image = image.rotate(180, expand=True)

    return image


def unit_vector(vector):
    if np.linalg.norm(vector) == 0:
        return None
    return vector / np.linalg.norm(vector)


def calc_angle(line):
    x1, y1, x2, y2 = line
    y1 = -y1
    y2 = -y2
    v1 = unit_vector(np.array([x2 - x1, y2 - y1]))
    v2 = np.array([0, 1])
    if v1 is None:
        return 0
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)) / np.pi * 180


@measure_timing
def find_best_angle(image):
    best_angle = 0
    max_variance = 0

    for angle in np.linspace(-2.0, 2.0, num=32):
        rotated = scipy.ndimage.rotate(image, angle)
        variance = np.var(np.sum(rotated, axis=1))

        if variance > max_variance:
            best_angle = angle
            max_variance = variance

    return best_angle


@measure_timing
def find_best_angle_threaded(image):
    variances = []
    threads = []

    for angle in np.linspace(-2.0, 2.0, num=32):
        thread = threading.Thread(
            target=cal_img_variance,
            args=(image, angle, variances)
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    best_angle = 0
    max_variance = 0
    for angle, variance in variances:
        if variance > max_variance:
            max_variance = variance
            best_angle = angle

    return best_angle


def cal_img_variance(image, angle, variances):
    rotated = scipy.ndimage.rotate(image, angle)
    variances.append((angle, np.var(np.sum(rotated, axis=1))))


def deskew_image(text_area):
    image, x, y, box = text_area

    if image.size == 0:
        return image

    height, width = image.shape
    cropped_height = width // 5

    angles_up = []
    angles_down = []
    start = 0
    while start < height:
        end = start + cropped_height
        if end > height:
            end = height
        img_segment = image[start:end]
        lines = cv2.HoughLinesP(
            img_segment,
            1,
            np.pi / 180,
            100,
            minLineLength=width // 3,
            maxLineGap=width // 30
        )
        start = end

        if lines is None:
            continue

        for line in lines:
            angle = calc_angle(line[0])
            if angle > 180 / 4 and angle < 90:
                angles_up.append(angle)
            if angle >= 90 and angle < 180 * (3 / 4):
                angles_down.append(angle)

    rotate_angle = 0
    if len(angles_up) > 0.8 * len(angles_down):
        rotate_angle = np.mean(angles_up)
    elif len(angles_down) > 0.8 * len(angles_up):
        rotate_angle = np.mean(angles_down)
    elif len(angles_up) == 0 and len(angles_down) == 0:
        rotate_angle = 90
    else:
        rotate_angle = np.mean(
            np.append(
                np.array(angles_up),
                np.array(angles_down)
            )
        )

    rotated = 0
    rotate_angle = rotate_angle - 90
    if math.fabs(rotate_angle) > 2:
        image = scipy.ndimage.rotate(image, rotate_angle)
        rotated += rotate_angle

        ih, iw = image.shape
        x = x - (iw - width) / 2
        y = y - (ih - height) / 2
        height = ih
        width = iw

    print('--------thread----------')
    best_angle = find_best_angle_threaded(image)
    print('angle: ', best_angle)
    # print('--------seq----------')
    # best_angle = find_best_angle(image)
    # print('angle: ', best_angle)

    image = scipy.ndimage.rotate(image, best_angle)
    rotated += best_angle
    cv2.imwrite('rotated.jpg', image)

    ih, iw = image.shape
    x = x - (iw - width) / 2
    y = y - (ih - height) / 2
    height = ih
    width = iw

    return (image, np.int0(x), np.int0(y), rotated)


def rotate_point(p, rotation, origin=(0, 0)):
    px, py = p
    ox, oy = origin
    rotation_in_rad = np.deg2rad(rotation)
    rotated_x = np.cos(rotation_in_rad) * (px - ox) -   \
        np.sin(rotation_in_rad) * (py - oy) + ox
    rotated_y = np.sin(rotation_in_rad) * (px - ox) +   \
        np.cos(rotation_in_rad) * (py - oy) + oy

    return np.array([np.int0(np.round(rotated_x)),
                     np.int0(np.round(rotated_y))])


def is_within(rect1, rect2):
    """returns True if rect1 is completely within rect2
    """
    is_left_side = True
    for pt in rect1:
        xp, yp = pt
        yp = -yp
        for i in range(4):
            x2, y2 = rect2[i]
            y2 = -y2
            x1, y1 = rect2[i + 1 if i + 1 < 4 else 0]
            y1 = -y1
            A = -(y2 - y1)
            B = x2 - x1
            C = -(A * x1 + B * y1)
            is_left_side = is_left_side and (A * xp + B * yp + C) >= 0

    return is_left_side


def extract_text_areas(image, original):
    edges = cv2.Canny(image, 50, 150)
    plt.imshow(edges, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.savefig('canny.jpg')

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3))
    dilation = cv2.dilate(edges, kernel, iterations=8)

    plt.imshow(dilation, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.savefig('dilate.jpg')

    img, cnts, __ = cv2.findContours(
        dilation,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cnt_boxes = [(cnt, np.int0(cv2.boxPoints(cv2.minAreaRect(cnt))))
                 for cnt in cnts]
    cnt_boxes.sort(key=lambda x: cv2.contourArea(x[1]), reverse=True)

    superset_boxes = [cnt_boxes[0]]
    for i in range(1, len(cnt_boxes)):
        box_tuple1 = cnt_boxes[i]
        box1_is_within = False
        for box_tuple2 in superset_boxes:
            if is_within(box_tuple1[1], box_tuple2[1]):
                box1_is_within = True
                break
        if not box1_is_within:
            superset_boxes.append(box_tuple1)

    copy = original.copy()
    for box in superset_boxes:
        cv2.drawContours(copy, [box[1]], 0, (0, 255, 0), 5)
    cv2.imwrite('contours.jpg', copy)

    text_areas = []
    for box in superset_boxes:
        x, y, w, h = cv2.boundingRect(box[1])

        if w < 50 or h < 50:
            continue

        if x < 0:
            x = 0
        if y < 0:
            y = 0
        crop = binarise_image_otsu(original[y: y + h, x: x + w])
        mask = np.zeros_like(crop)
        contours = np.add(box[0], np.array([-x, -y]))
        cv2.drawContours(mask, [contours], 0, 255, -1)
        masked_image = cv2.bitwise_and(crop, mask)
        text_areas.append((masked_image, x, y, box[1]))

    for i, text in enumerate(text_areas):
        cv2.imwrite('contours{}.jpg'.format(i), text[0])

    return text_areas


def binarise_image(image):
    image = gray_and_blur(image)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 2)
    image = cv2.bitwise_not(image)
    kernel = np.ones((5, 5), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.dilate(image, kernel, iterations=1)
    cv2.imwrite('bin.jpg', image)

    return image


def binarise_image_otsu(image):
    image = gray_and_blur(image)
    __, img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return cv2.bitwise_not(img)


def gray_and_blur(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (5, 5), 0)

    return image


def evaluate_text_area(text_area, results):
    if text_area[0].size == 0:
        return

    text, tx, ty, angle = deskew_image(text_area)
    image = cv2.bitwise_not(text)

    image = Image.fromarray(image)

    with PyTessBaseAPI(psm=PSM.SINGLE_BLOCK, lang='eng') as api:
        api.SetImage(image)
        boxes = api.GetComponentImages(RIL.TEXTLINE, True)
        for im, box, __, __ in boxes:
            x = box['x']
            y = box['y']
            w = box['w']
            h = box['h']

            api.SetRectangle(x, y, w, h)
            ocrResult = api.GetUTF8Text()

            iw, ih = image.size
            origin = (iw / 2, ih / 2)

            detected_area = np.array([
                rotate_point((x, y), angle, origin=origin),
                rotate_point((x, y + h), angle, origin=origin),
                rotate_point((x + w, y + h), angle, origin=origin),
                rotate_point((x + w, y), angle, origin=origin),
            ])

            detected_area = np.add(detected_area, np.array([tx, ty]))
            result = (ocrResult, detected_area)
            results.put(result)


if __name__ == '__main__':
    image = None
    img_name = './test/cropped/test15'
    try:
        img_path = img_name + '.JPG'
        image = Image.open(img_path)
    except Exception:
        img_path = img_name + '.jpg'
        image = Image.open(img_path)

    with PyTessBaseAPI() as api:
        image = rotate_to_upright(image, api)

    with PyTessBaseAPI(psm=PSM.SINGLE_BLOCK, lang='eng') as api:
        api.SetImage(image)
        original = np.array(image)

        binarised = binarise_image(original)
        # binarised = np.array(api.GetThresholdedImage())
        # binarised = cv2.bitwise_not(binarised)
        # cv2.imwrite('binarised.jpg', binarised)

        text_areas = extract_text_areas(binarised, original)

        results = Queue()
        processes = []
        for text_area in text_areas:
            process = Process(
                target=evaluate_text_area,
                args=(text_area, results)
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        while not results.empty():
            result = results.get()
            print(result[0])
            cv2.drawContours(original, [result[1]], 0, (0, 255, 0), 5)

        cv2.imwrite('result.jpg', original)

    print('end')
