# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 11:24:15 2021

@author: Christian Pfister
https://cpfister.com
https://github.com/christianpfister43?tab=repositories

Schuldenuhr: https://www.gold.de/staatsverschuldung-deutschland/
"""

import numpy as np
from PIL import ImageGrab
import cv2
import os

#%% set your custom paths and parameters here!
"""
parameters here work for me for:
https://www.gold.de/staatsverschuldung-deutschland/
google chrome
res: 1920 * 1080
you will need to adapt these for your system, and problem
"""
im_path = './data'
number_of_digits = 13 # German debt hase currently 13 digits!
# width and height of the digits
w = 20
h = 28
# offset where the digits begin on the screen
x_0 = 800
y_0 = 613

# width of the sliding window
delta_x = w
# width of the "." that seperates bundels of 3 digits
dot_width = 11



#%% loop over all digits
digit_array = [[] for n in range(number_of_digits)]
dot_offset = 0
for n in range (number_of_digits):
    if (n==1)|(n==4)|(n==7)|(n==10):
        dot_offset+= dot_width        # to jump over "." every 3 digits
        # depending on your problem this will need adaption, e.g. jump over a "." and a "," can be different

    # grab image from screen and transform to numpy array
    screen_cap = ImageGrab.grab(bbox=(x_0+n*delta_x+dot_offset,y_0,x_0+n*delta_x+w+dot_offset,y_0+h))
    screen_cap_array = np.array(screen_cap.getdata(), dtype='uint8')\
        .reshape((screen_cap.size[1],screen_cap.size[0],3))
    # cv2.imshow(f'window_{n}', printscreen_numpy)
    
    # padding the image-border with white and resize to 28*28 pixels
    # this helped me for recognizing the digits later
    im = cv2.copyMakeBorder(screen_cap_array.copy(),3,3,3,3,cv2.BORDER_CONSTANT,value=[255,255,255])
    im = cv2.resize(im, (28,28))
    # save image of digit
    cv2.imwrite(f'{im_path}/digit_{n}.png',im)
