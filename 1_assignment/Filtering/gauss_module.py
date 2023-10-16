# import packages: numpy, math (you might need pi for gaussian functions)
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv2


"""
Gaussian function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian values Gx computed at the indexes x
"""
def gauss(sigma):

    y = round(sigma) # Rounding the value of sigma to the nearest integer
    x = np.arange(-3*y, 3*y+1) # Interval for which the gaussian filter is defined
    Gx = ((1. / (np.sqrt(2 * np.pi) * sigma)) * np.exp(((-x ** 2) / (2 * sigma ** 2)))) # Calculating the actual filter using the definition

    return Gx, x


"""
Implement a 2D Gaussian filter, leveraging the previous gauss.
Implement the filter from scratch or leverage the convolve2D method (scipy.signal)
Leverage the separability of Gaussian filtering
Input: image, sigma (standard deviation)
Output: smoothed image
"""
def gaussianfilter(img, sigma):
    Gx = gauss(sigma)[0] # Just need the filter, which is returned at [0]
    Gx_y = Gx.reshape(1, Gx.size) # Need a vertical vector, still 2D
    Gx_x = Gx_y.T
    
    smooth_img = conv2(conv2(img, Gx_y, mode='same'), Gx_x, mode='same') # Applying 2 convolutions separately, making use of the separability property

    return smooth_img



"""
Gaussian derivative function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian derivative values Dx computed at the indexes x
"""
def gaussdx(sigma):

    y = round(sigma) # Rounding the value of sigma to the nearest integer
    x = np.arange(-3*sigma, 3*sigma + 1) # Definining the interval for which the gaussian derivative filter is defined
    Dx = ((-1. / (np.sqrt(2 * np.pi) * sigma**3))* x * np.exp(((-x ** 2) / (2 * sigma ** 2)))) # Applying the definition

    return Dx, x



def gaussderiv(img, sigma):
    Gx = gaussdx(sigma)[0] # Just need the gaussian derivative filter which is at [0]
    Dy = Gx.reshape(1, Gx.size) # Vertical vector 
    Dx = Dy.T # Horizontal vector
    
    imgDx = conv2(img, Dx, mode='same')
    imgDy = conv2(img, Dy, mode='same')
    
    return imgDx, imgDy
    

