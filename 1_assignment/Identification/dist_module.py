import numpy as np
import math



# Compute the intersection distance between histograms x and y
# Return 1 - hist_intersection, so smaller values correspond to more similar histograms
# Check that the distance range in [0,1]

def dist_intersect(x,y):
    
    #... (your code here)
    min_vi_qi_sum = np.sum(np.minimum(x,y)) # np.minimum gives us the pairwise minimum in an array
    q_sum = np.sum(x)
    v_sum = np.sum(y)
    dist = 0.5 * ((min_vi_qi_sum / q_sum + min_vi_qi_sum / v_sum))
    
    assert dist >= 0 and dist <= 1 # So we expect values to be normalized.
    
    return 1 - dist


# Compute the L2 distance between x and y histograms
# Check that the distance range in [0,sqrt(2)]

def dist_l2(x,y):
    
    #... (your code here)
    dist = np.sum((x-y)**2)
    assert dist >= 0 and dist <= np.sqrt(2) # Same as before, values need to be normalized.
    return dist


# Compute chi2 distance between x and y
# Check that the distance range in [0,Inf]
# Add a minimum score to each cell of the histograms (e.g. 1) to avoid division by 0

def dist_chi2(x,y):
    
    #... (your code here)
    epsilon = 1
    dist = np.sum( ((x-y) ** 2)/ (x + y + epsilon))
    assert dist >= 0 # Even the x,y should be normalized, we have no way of checking that, due to the denominator
    return dist

def get_dist_by_name(x, y, dist_name):
  if dist_name == 'chi2':
    return dist_chi2(x,y)
  elif dist_name == 'intersect':
    return dist_intersect(x,y)
  elif dist_name == 'l2':
    return dist_l2(x,y)
  else:
    assert False, 'unknown distance: %s'%dist_name
  




