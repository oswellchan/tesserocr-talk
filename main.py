from PIL import Image
from tesserocr import PyTessBaseAPI, RIL, PSM, Orientation
import time
import copy
import numpy as np


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


img_path = './test/cropped/test16.jpg'
image = Image.open(img_path)

# with PyTessBaseAPI() as api:
#     image = rotate_to_upright(image, api)
#     api.SetImage(image)
#     binarised = api.GetThresholdedImage()
#     matrix = get_weighted_img(binarised, 1)

with PyTessBaseAPI() as api:
    image = rotate_to_upright(image, api)

with PyTessBaseAPI(psm=PSM.SINGLE_BLOCK) as api:
    api.SetImage(image)
    width, height = image.size
    names_coord, prices_coord = split_image(api.GetThresholdedImage())
    names_img = image.crop((names_coord[0], 0, names_coord[1], height - 1))
    prices_img = image.crop((prices_coord[0], 0, prices_coord[1], height - 1))

    api.SetPageSegMode(PSM.SINGLE_LINE)
    api.SetImage(names_img)
    boxes = api.GetComponentImages(RIL.TEXTLINE, True)

    for im, box, __, __ in boxes:
        api.SetRectangle(box['x'], box['y'], box['w'], box['h'])
        ocrResult = api.GetUTF8Text()
        conf = api.MeanTextConf()
        print(ocrResult)
    
    draw_boxes(boxes, image)
               


        

