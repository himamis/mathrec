from utilities import parse_arg
from file_utils import list_files, read_img
import cv2
import numpy as np


def remove_shadow(image):
    dilated_img = cv2.dilate(image, np.ones((7, 7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 35)
    diff_img = 255 - cv2.absdiff(image, bg_img)
    #norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return image

def threshold_image(image):
    cv2.normalize()
    ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 2)
    #image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 20)
    return image


def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = remove_shadow(image)
    #image = cv2.GaussianBlur(image, (5,5), 0)
    #image = threshold_image(image)
    return image

images_dir = parse_arg("--images", "/Users/balazs/university/mathrec/image_preprocessor/images")
images = []

for image_file in list_files(images_dir):
    if image_file.endswith("jpg"):
        image = read_img(image_file)
        processed_image = preprocess_image(image)
        images.append(processed_image)


width = 200
height = 100
margin = 50
for i, image in enumerate(images):
    window_name = "window" + str(i)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    cv2.moveWindow(window_name, i % 4 * (width + margin), int((i / 4)) * (height + margin))
    cv2.imshow(window_name, image)
cv2.waitKey(0)
cv2.destroyAllWindows()