# tesserocr-talk
This repo contains the resources used in [PyData SG Feb 2017 talk](https://www.meetup.com/PyData-SG/events/235981769/).

It contains:

1. Presentation slides `Image to Text.pdf`
2. Code sample `main.py`
3. Images used to run the code sample `test/cropped`

## Setup

Requires:

* libtesseract (>=3.04)
* libleptonica (>=1.71)

On Debian/Ubuntu:

    $ apt-get install tesseract-ocr libtesseract-dev libleptonica-dev

Alternatively, you can compile [tesseract](https://github.com/tesseract-ocr/tesseract/wiki/Compiling) and [leptonica](http://www.leptonica.org/source/README.html) manually.

It also requires some python packages:

* [Cython](http://cython.org/)
* [tesserocr](https://github.com/sirfz/tesserocr)
* [Pillow](http://python-pillow.org/)
* [NumPy](http://www.numpy.org/)
* [opencv-python](http://docs.opencv.org/3.2.0/d5/de5/tutorial_py_setup_in_windows.html)
* [SciPy](https://www.scipy.org)

## Running the code
Run the script with

    $ python main.py <file_path>

It generates `result.jpg` that shows the results of OCR and the bounding boxes of identified text area.
