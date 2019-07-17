# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:43:44 2019

@author: William Wakefield
"""
import cv2
import numpy as np
#import matplotlib.pyplot as plt

"""
Canny Function
Converts the input image to grayscale, and then runs
the image through the Canny functino, which 
includes the GaussianBlur function. 
The GaussianBlur function call is redudancy. 

"""
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5),0)  
    canny = cv2.Canny(blur, 50, 150)
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2= line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10) #line(image, first points, end points, tuple of RGB color of line, pixel thickness) 
    return line_image
       
def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
    [(200, height), (1100, height), (550,250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask) #takes original image and masks them based on mask
    return masked_image
    
image = cv2.imread('White-Broken-1.jpg')
lane_image = np.copy(image)
canny = canny(lane_image)
cropped_image = region_of_interest(canny)
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) #2nd and 3rd args represent size of bins for creating line of best fit from Hough space. 4th argument is threshold; 
#Threshold is minimum number of votes/intersections needed to detect a line; 5th artgument is placceholder array (empty); 
line_image =  display_lines(lane_image, lines)
combined_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
cv2.imshow('result', line_image)
cv2.imshow('Final Result', combined_image)
cv2.imshow("lol", canny)
cv2.waitKey(0)


