import numpy as np
from numpy import histogram as hist



#Add the Filtering folder, to import the gauss_module.py file, where gaussderiv is defined (needed for dxdy_hist)
import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
filteringpath = os.path.join(parentdir, 'Filtering')
sys.path.insert(0,filteringpath)
import gauss_module



#  compute histogram of image intensities, histogram should be normalized so that sum of all values equals 1
#  assume that image intensity varies between 0 and 255
#
#  img_gray - input image in grayscale format
#  num_bins - number of bins in the histogram
def normalized_hist(img_gray, num_bins):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'


    #... (your code here)
    image_vector = img_gray.reshape(img_gray.size) # Transform the image into a vector, we'll iterate over it more easily
    hists = np.zeros(num_bins) # Initialize each bin in the histogram to 0
    bin_width = 255 / num_bins # Get the bin span (width)
    
    for pixel in image_vector: # Iterate over each pixel
        bin_index = int(np.floor(pixel / bin_width)) # Get the index of the bin (0 to num_bins - 1)
        hists[bin_index] += 1 # Increment the bin value (this pixel belongs here! Count it)
        
    hists /= len(image_vector) # Normalize by dividing by the number of counted occurrences (the number of pixels)
    bins = np.array([i * bin_width for i in range(num_bins + 1)])
    
    assert sum(hists) == 1
    
    return hists, bins



#  Compute the *joint* histogram for each color channel in the image
#  The histogram should be normalized so that sum of all values equals 1
#  Assume that values in each channel vary between 0 and 255
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^3
#
#  E.g. hists[0,9,5] contains the number of image_color pixels such that:
#       - their R values fall in bin 0
#       - their G values fall in bin 9
#       - their B values fall in bin 5
def rgb_hist(img_color_double, num_bins):
    assert len(img_color_double.shape) == 3, 'image dimension mismatch'
    assert img_color_double.dtype == 'float', 'incorrect image type'

    #... (your code here)
    image_vector = img_color_double.reshape(-1, 3) # Transform into list of RGB tuples
    bin_width = 255 / num_bins
    
    
    #Define a 3D histogram  with "num_bins^3" number of entries
    hists = np.zeros((num_bins, num_bins, num_bins))
    
    # Loop for each pixel i in the image 
    for i in range(img_color_double.shape[0]*img_color_double.shape[1]):
        # Increment the histogram bin which corresponds to the R,G,B value of the pixel i
        #... (your code here)
        r,g,b = image_vector[i][0],image_vector[i][1],image_vector[i][2]
        r_bin_index = int(np.floor(r / bin_width))
        g_bin_index = int(np.floor(g / bin_width))
        b_bin_index = int(np.floor(b / bin_width))
        
        hists[r_bin_index,g_bin_index,b_bin_index] += 1
        pass


    #Normalize the histogram such that its integral (sum) is equal 1
    #... (your code here)
    hists /= (img_color_double.shape[0]*img_color_double.shape[1])

    #Return the histogram as a 1D vector
    hists = hists.reshape(hists.size)

    return hists



#  Compute the *joint* histogram for the R and G color channels in the image
#  The histogram should be normalized so that sum of all values equals 1
#  Assume that values in each channel vary between 0 and 255
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^2
#
#  E.g. hists[0,9] contains the number of image_color pixels such that:
#       - their R values fall in bin 0
#       - their G values fall in bin 9
def rg_hist(img_color_double, num_bins):
    assert len(img_color_double.shape) == 3, 'image dimension mismatch'
    assert img_color_double.dtype == 'float', 'incorrect image type'


    #... (your code here)
    image_vector = img_color_double.reshape(-1, 3) # Transform into list of RGB tuples
    bin_width = 255 / num_bins
    
    
    #Define a 2D histogram  with "num_bins^2" number of entries
    hists = np.zeros((num_bins, num_bins))
    
    
    #... (your code here)
    for color_tuple in image_vector:
        r,g = color_tuple[0],color_tuple[1]
        r_bin_index = int(np.floor(r / bin_width))
        g_bin_index = int(np.floor(g / bin_width))
        
        hists[r_bin_index,g_bin_index] += 1
        pass



    #Return the histogram as a 1D vector
    hists = hists.reshape(hists.size)
    
    hists /= (len(image_vector))
    
    assert sum(hists) == 1

    return hists




#  Compute the *joint* histogram of Gaussian partial derivatives of the image in x and y direction
#  Set sigma to 3.0 and cap the range of derivative values is in the range [-6, 6]
#  The histogram should be normalized so that sum of all values equals 1
#
#  img_gray - input gray value image
#  num_bins - number of bins used to discretize each dimension, total number of bins in the histogram should be num_bins^2
#
#  Note: you may use the function gaussderiv from the Filtering exercise (gauss_module.py)
def dxdy_hist(img_gray, num_bins):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'


    #... (your code here)
    sigma = 3.0

    img_dx, img_dy = gauss_module.gaussderiv(img_gray, sigma)
    np.clip(img_dx, -6, 6, out=img_dx) # If the value is greater than 6, set it to 6
    np.clip(img_dy, -6, 6, out=img_dy) # If the value is greater than 6, set it to 6
    
    img_dx = img_dx.reshape(img_dx.size)
    img_dy = img_dy.reshape(img_dy.size)
    
    bin_width = 13 / num_bins # 12 because later we'll shift the -6 to 6 values of the derivative to 0 to 12 (13 discrete values)

    #Define a 2D histogram  with "num_bins^2" number of entries
    hists = np.zeros((num_bins, num_bins))

    #... (your code here)
    for edx, edy in zip(img_dx, img_dy):
        bin_index_dx = int(np.floor((edx + 6)/bin_width))
        bin_index_dy = int(np.floor((edy + 6)/bin_width))
        
        hists[bin_index_dx,bin_index_dy] += 1 # Count one more occurrence of a certain tuple into hists
        
    hists /= len(img_dx)

    #Return the histogram as a 1D vector
    hists = hists.reshape(hists.size)

    assert sum(hists) == 1

    return hists



def is_grayvalue_hist(hist_name):
  if hist_name == 'grayvalue' or hist_name == 'dxdy':
    return True
  elif hist_name == 'rgb' or hist_name == 'rg':
    return False
  else:
    assert False, 'unknown histogram type'


def get_hist_by_name(img, num_bins_gray, hist_name):
  if hist_name == 'grayvalue':
    return normalized_hist(img, num_bins_gray)
  elif hist_name == 'rgb':
    return rgb_hist(img, num_bins_gray)
  elif hist_name == 'rg':
    return rg_hist(img, num_bins_gray)
  elif hist_name == 'dxdy':
    return dxdy_hist(img, num_bins_gray)
  else:
    assert False, 'unknown distance: %s'%hist_name

