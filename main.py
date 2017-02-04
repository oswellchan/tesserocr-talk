from PIL import Image
from tesserocr import PyTessBaseAPI, RIL, PSM, Orientation
import time
import copy
import numpy as np
import cv2
import math
import scipy.ndimage
import matplotlib.pyplot as plt

THRESHOLD_WHITE = 200
WEIGHT = 5

class ImagePixels:
    """docstring for ImagePixels"""
    def __init__(self, image):
        self.width, self.height = image.size
        self.pixels = list(image.getdata())
        self.mode = copy.copy(image.mode)

    def __getitem__(self, coords):
        x, y = coords
        return self.pixels[x + y * self.width]

    def __setitem__(self, coords, value):
        x, y = coords
        self.pixels[x + y * self.width] = value

    def save_image(self, file_name):
        img = Image.new(self.mode, (self.width, self.height))
        img.putdata(self.pixels)
        img.save(file_name)

def measure_timing(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(time.time() - start)
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

def draw_bw_box(boxes, image, file_name='output.jpg', color=0):
    img = image.copy()
    pix = img.load()

    width, height = img.size
    for box in boxes:
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

def draw_hort_lines(lines, image, file_name='output.jpg'):
    img = image.copy()
    pix = img.load()
    width, __ = image.size
    for upper, lower in lines:
        for i in range(width):
            pix[i, upper] = (255, 0, 0)
            pix[i, lower] = (255, 0, 0)

    img.save(file_name)

def draw_vert_lines(lines, image, file_name='output.jpg', color=(255, 0, 0)):
    img = image.copy()
    pix = img.load()
    __, height = image.size
    for left, right in lines:
        for i in range(height):
            pix[left, i] = color
            pix[right, i] = color

    img.save(file_name)
    print('done')

@measure_timing
def split_image(image, threshold=THRESHOLD_WHITE, col_threshold_factor=0.1):
    width, height = image.size
    pix = image.load()

    whitespaces = []
    max_ws_index = 0
    max_whitespace = 0
    last_black = width - 1
    for x in range(width - 1, 5*width//8, -1):
        column = 0
        for y in range(height):
            if pix[x, y] < threshold:
                column += 1

        if (column < height * col_threshold_factor and
                x > 5*width//8 + 1):
            continue
        
        whitespace_len = last_black - x
        whitespaces.append((x, last_black))
        last_black = x

        if whitespace_len > max_whitespace:
            max_ws_index = len(whitespaces) - 1
            max_whitespace = whitespace_len

    max_start, max_end = whitespaces[max_ws_index]
    left_padding = 0
    if max_ws_index < len(whitespaces) - 1:
        __, end = whitespaces[max_ws_index + 1]
        separation = max_start - end
        if separation > 0.05 * width:
            left_padding = 0.05 * width
        else:
            left_padding = 0.7 * separation

    right_padding = 0
    if max_ws_index > 0:
        start, __ = whitespaces[max_ws_index - 1]
        separation = start - max_end
        if separation > 0.05 * width:
            right_padding = 0.05 * width
        else:
            right_padding = 0.7 * separation

    #draw_vert_lines(whitespaces, image, file_name='output1.jpg', color=0)

    return (0, max_start - left_padding), (max_end + right_padding , width - 1)

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

    remove_black_border(image)

    return image

def remove_black_border(image):
    pass

def get_weighted_img(image, radius):
    w, h = image.size
    pix = image.load()

    return [[get_weight(x, y, w, h, radius, pix) for x in range(w)] for y in range(h)]
            

def get_weight(x, y, w, h, radius, pix):
    if pix[x, y] > THRESHOLD_WHITE:
        return 0

    result = 1
    for i in range(x - radius, x + radius):
       for j in range(y - radius, y + radius):
            if (0 <= i and i < w
                    and 0 <= j and j < h):
                result += WEIGHT  # maybe do exponential

    return result

def unit_vector(vector):
    if np.linalg.norm(vector) == 0:
        return None
    return vector / np.linalg.norm(vector)

def calc_angle(line):
    x1, y1, x2, y2 = line
    v1 = unit_vector(np.array([x1 - x2, y1 - y2]))
    v2 = np.array([0, 1])
    if v1 is None:
        return 0
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))/np.pi * 180

def find_best_angle(image):
    best_angle = 0
    max_variance = 0
    fig_num = 0
    count = 0
    for angle in np.linspace(-2.0, 2.0, num=32):
        fig_num += 1
        plt.subplot(2, 3, fig_num)
        rotated = scipy.ndimage.rotate(image, angle)
        variance = np.var(np.sum(rotated, axis=1))
        plt.imshow(rotated)
        plt.title(variance)
        if fig_num == 6:
            fig_num = 0
            count += 1
            plt.savefig('all_angles{}.jpg'.format(count))

        if variance > max_variance:
            best_angle = angle
            max_variance = variance

    return best_angle

def calculate_variance(image):
    return 

if __name__ == '__main__':
    image = None
    img_name = './test/cropped/test19'
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
        width, height = image.size
        binarised = np.array(api.GetThresholdedImage())
        binarised = cv2.bitwise_not(binarised)

        cropped_height = width // 5
        all_lines = np.array([[[0, 0, 0, 0]]])
        start = 0
        while start < height:
            end = start + cropped_height
            if end > height:
                end = height
            img_segment = binarised[start:end]
            lines = cv2.HoughLinesP(
                img_segment,
                1,
                np.pi/180,
                100,
                minLineLength=width//3,
                maxLineGap=width//20
            )

            if lines is not None:
                lines += np.array([[0, start, 0, start]])
                all_lines = np.append(all_lines, lines, axis=0)

            start = end

        # for line in all_lines:
        #     x1,y1,x2,y2 = line[0]
        #     cv2.line(binarised,(x1,y1),(x2,y2),(255,0,0),2)

        # cv2.imwrite('lines.jpg', binarised)

        angles_up = []
        angles_down = []
        for line in all_lines:
            angle = calc_angle(line[0])
            if angle > 180/4 and angle < 90:
                angles_up.append(angle)
            if angle >= 90 and angle < 180 * (3/4):
                angles_down.append(angle)

        rotate_angle = 0
        if len(angles_up) > 0.8 * len(angles_down):
            rotate_angle = np.mean(angles_up)
        elif len(angles_down) > 0.8 * len(angles_up):
            rotate_angle = np.mean(angles_down)
        else:
            rotate_angle = np.mean(
                np.append(
                    np.array(angles_up),
                    np.array(angles_down)
                )
            )

        rotate_angle = rotate_angle - 90
        if math.fabs(rotate_angle) > 2:
            binarised = scipy.ndimage.rotate(binarised, rotate_angle)

        best_angle = find_best_angle(binarised)
        binarised = scipy.ndimage.rotate(binarised, best_angle)
        cv2.imwrite('rotated.jpg', binarised)

        binarised = cv2.bitwise_not(binarised)

        # print(rotate_angle - 90)
        # image = image.rotate(rotate_angle - 90)
        # api.SetImage(image)

        # names_coord, prices_coord = split_image(api.GetThresholdedImage())
        # names_img = image.crop((names_coord[0], 0, names_coord[1], height - 1))
        # prices_img = image.crop((prices_coord[0], 0, prices_coord[1], height - 1))

        # api.SetPageSegMode(PSM.SINGLE_LINE)
        # api.SetImage(names_img)
        
        image = Image.fromarray(binarised)
        api.SetImage(image)
        image = api.GetThresholdedImage()
        image.save('bina.jpg')
        boxes = api.GetComponentImages(RIL.TEXTLINE, True)

        for im, box, __, __ in boxes:
            api.SetRectangle(box['x'], box['y'], box['w'], box['h'])
            ocrResult = api.GetUTF8Text()
            conf = api.MeanTextConf()
            print(ocrResult)
        
        draw_boxes(boxes, image, color=0)
