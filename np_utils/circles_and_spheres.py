#!/usr/bin/env python
'''Create the first 100 circles and the first 20 spheres so we don't have to recalculate them if they are small!
Relatively negligible amount of memory (364450 total points, about 3MB on a 64-bit computer)'''
from __future__ import absolute_import
from builtins import range

from .array_drawing import ImageCircle,ImageSphere

circs = [None]+[ImageCircle(i) for i in range(1,100)]
shprs = [None]+[ImageSphere(i) for i in range(1,20)]
