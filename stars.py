import numpy as np
import mahotas as mh
from skimage import measure

# Load your image (e.g., 'sky.jpeg')
sky = mh.imread('sky.jpeg')

# Threshold the image using Otsu's method
t = mh.thresholding.otsu(sky.astype('uint8'))

# Label connected components (stars) in the binary image
labeled, stars = mh.label(sky > t)

# Now 'stars' contains the labeled regions (each star)
# You can count the number of stars using 'stars.max()'
num_stars = stars.max()

print(f"Number of stars in the image: {num_stars}")
