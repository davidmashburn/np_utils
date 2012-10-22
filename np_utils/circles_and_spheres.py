#!/usr/bin/env python
'''Create the first 100 circles and the first 20 spheres so we don't have to recalculate them if they are small!
Relatively negligible amount of memory (364450 total points, about 3MB on a 64-bit computer)'''

from np_utils import ImageCircle,ImageSphere

circs = [ImageCircle(i) for i in range(1,100)]
shprs = [ImageSphere(i) for i in range(1,20)]

