import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import histogram_module
import dist_module

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray



# model_images - list of file names of model images
# query_images - list of file names of query images
#
# dist_type - string which specifies distance type:  'chi2', 'l2', 'intersect'
# hist_type - string which specifies histogram type:  'grayvalue', 'dxdy', 'rgb', 'rg'
#
# note: use functions 'get_dist_by_name', 'get_hist_by_name' and 'is_grayvalue_hist' to obtain 
#       handles to distance and histogram functions, and to find out whether histogram function 
#       expects grayvalue or color image

def find_best_match(model_images, query_images, dist_type, hist_type, num_bins):

    hist_isgray = histogram_module.is_grayvalue_hist(hist_type)
    model_hists = compute_histograms(model_images, hist_type, hist_isgray, num_bins) # get signal from image file
    query_hists = compute_histograms(query_images, hist_type, hist_isgray, num_bins) # get signal from image file

    D = np.zeros((len(model_images), len(query_images))) # matrix of distances, initialized with all zero values
    

    #... (your code here)
    best_match = np.zeros(len(query_images)) # vector of best matches, initialized with zeros
    
    for i in range(len(query_images)):
        for j in range(len(model_images)):
            
            query_hist = query_hists[i]
            model_hist = model_hists[j]
            
            distance = dist_module.get_dist_by_name(model_hist, query_hist, dist_type)
            
            D[j, i] = distance
        
        best_match[i] = D[: , i].argmin() # Best match will contain indices, get the one from the distance matrix for an image i such that dist is min
    
    return best_match, D



def compute_histograms(image_list, hist_type, hist_isgray, num_bins):
    
    image_hist = []

    # Compute hisgoram for each image and add it at the bottom of image_hist
    #... (your code here)
    for image_path in image_list:
        img_color = np.array(Image.open(image_path)) # First put the image in a np.array, supposing it's RGB.
        
        if hist_isgray: # If the histogram type requires greyscale..
            image = rgb2gray(img_color.astype('float')) # .. convert the image to greyscale
        else:
            image = img_color.astype('float') # Otherwise keep it as it is.
        
        if hist_type == 'grayvalue':
            hist = histogram_module.get_hist_by_name(image, num_bins, hist_type)[0] #ormalized_hist returns two arguments! just need the first one
        else:
            hist = histogram_module.get_hist_by_name(image, num_bins, hist_type)
            
        image_hist.append(hist) # Append the histogram to our list.
    
    return image_hist

# img_color = np.array(Image.open('./model/obj100__0.png'))
# img_gray = rgb2gray(img_color.astype('double'))


# For each image file from 'query_images' find and visualize the 5 nearest images from 'model_image'.
#
# Note: use the previously implemented function 'find_best_match'
# Note: use subplot command to show all the images in the same Python figure, one row per query image

def show_neighbors(model_images, query_images, dist_type, hist_type, num_bins):
    
    
    plt.figure()

    num_nearest = 5  # show the top-5 neighbors
    
    #... (your code here)
    fig, ax = plt.subplots(len(query_images), num_nearest + 1)
    fig.suptitle(str('KNN with %s distance, %d bins and %s hist type.' % (dist_type, num_bins, hist_type)))
    fig.set_figwidth(12) # The default image size is a little small..
    fig.set_figheight(10) # Best if we changed it.
    
    D = find_best_match(model_images, query_images, dist_type, hist_type, num_bins)[1] # returns only the vector of distances

    for i in range(len(query_images)):
        ax[i][0].imshow(np.array(Image.open(query_images[i]))) # show the query image at row i, position 0 (leftmost)
        ax[i][0].set_title('Q'+str(i)) # Give the image a name
        ax[i][0].get_xaxis().set_visible(False) # Hide xticks
        ax[i][0].get_yaxis().set_visible(False) # Hide yticks, we don't really need that (it's just pixel indices)
        
        
        distances_from_ith_img = D[:, i]        
        
        knn_indices = distances_from_ith_img.argsort()[:num_nearest] # Get the indices of the k (num_nearest) images, sorted by the distance.
        
        for j,nn_idx in zip(range(len(knn_indices)), knn_indices):
            ax[i][j+1].imshow(np.array(Image.open(model_images[nn_idx]))) # Show the image at ith row, position 1 + [0, num_nearest]
            ax[i][j+1].set_title('M'+ '%.2f' %(np.around(distances_from_ith_img[nn_idx], decimals=2))) # Reference image from the professor
            ax[i][j+1].get_xaxis().set_visible(False) # Again, make xticks
            ax[i][j+1].get_yaxis().set_visible(False) # and y ticks invisible.
        
