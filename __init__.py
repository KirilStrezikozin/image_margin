"""
Image Margin
============

This module comes with a set of functions designed to create margin for an
image texture. Margin (also called dilation or padding) is a post-processing
effect, which is used to extend the borders of UV islands on an image texture
to fill the nearby empty pixels with similar colors.

Provides (each includes solutions in Python, C, or interfaced C via a Python
module):
    1. Simple margin creation with a fixed number of pixels to extend for.
    2. Infinite margin implementations with supported multiprocessing.

Limitations:
    1. No interpolation between adjacent pixels.
    2. Does not depend on UV data (e.g. islands boundaries).
    3. No image I/O, accepts flattened one-dimensional array of pixels.
    4. Computations are handled on CPU.

Usage
-----

1. TODO

Copyright (c) 2023 kemplerart
"""


def hello():
    print("hello")
