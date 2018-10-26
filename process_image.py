from skimage import img_as_ubyte
from skimage.io import imread
from skimage.filters import gaussian, threshold_minimum
from skimage.morphology import square, erosion, thin

import numpy as np
import cv2

# image.py

# dependencies ...

# Process requested image
def binarize(image_abs_path):

    # Convert color image (3-channel deep) into grayscale (1-channel deep)
    # We reduce image dimensionality in order to remove unrelevant features like color.
    grayscale_img = imread(image_abs_path, as_grey=True)

    # Apply Gaussian Blur effect - this removes image noise
    gaussian_blur = gaussian(grayscale_img, sigma=1)

    # Apply minimum threshold
    thresh_sauvola = threshold_minimum(gaussian_blur)

    # Convert thresh_sauvola array values to either 1 or 0 (white or black)
    binary_img = gaussian_blur > thresh_sauvola

    return binary_img

# image.py

# dependencies ...

# def process(image_abs_path) ...

def shift(contour):

    # Get minimal X and Y coordinates
    x_min, y_min = contour.min(axis=0)[0]

    # Subtract (x_min, y_min) from every contour point
    return np.subtract(contour, [x_min, y_min])

def get_scale(cont_width, cont_height, box_size):

    ratio = cont_width / cont_height

    if ratio < 1.0:
        return box_size / cont_height
    else:
        return box_size / cont_width

def extract_patterns(image_abs_path):

    #max_intensity = 1
    # Here we define the size of the square box that will contain a single pattern
    #box_size = 32

    binary_img = binarize(image_abs_path)

    # Apply erosion step - make patterns thicker
    #eroded_img = erosion(binary_img, selem=square(3))

    # Inverse colors: black --> white | white --> black
    #binary_inv_img = max_intensity - eroded_img

    # Apply thinning algorithm
    #thinned_img = thin(binary_inv_img)

    # Before we apply opencv method, we need to convert scikit image to opencv image
    #thinned_img_cv = img_as_ubyte(binary_inv_img)

    # Find contours
    # _, contours, _ = cv2.findContours(thinned_img_cv, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours from left to right (sort by bounding rectangle's X coordinate)
    # contours = sorted(contours, key=lambda cont: cv2.boundingRect(cont)[0])

    # Initialize patterns array
    # patterns = []

    return img_as_ubyte(binary_img)

patterns = extract_patterns("/Users/balazs/Downloads/IMG_1978.jpg")
#for pattern in patterns:
cv2.imshow("csa", patterns)
cv2.waitKey(0)