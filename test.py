import tesserocr
from PIL import Image

print(tesserocr.tesseract_version())  # print tesseract-ocr version
print(tesserocr.get_languages())  # prints tessdata path and list of available languages

image = Image.open('output0.jpg')
print(tesserocr.image_to_text(image))  # print ocr text from image #