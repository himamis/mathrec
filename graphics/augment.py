import file_utils
from graphics.utils import *

def _rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), borderValue=(255, 255, 255))


class Augmentor:

    def __init__(self, base):
        self.background_files = [x for x in file_utils.list_files(base) if x.endswith(".png")]

    def augment(self, img):
        new_image = self.rotate(img)
        new_image = self.background(new_image)
        return self.blur(new_image)

    def rotate(self, img):
        angle = np.random.random_integers(-5, 5)
        return _rotate_bound(img, angle)

    def blur(self, img):
        num = np.random.random_integers(0, 2) * 2 + 1
        sigma = np.random.random_integers(1, 5)
        return cv2.GaussianBlur(img, (num, num), sigma)

    def background(self, img):
        index = np.random.random_integers(0, len(self.background_files) - 1)
        bckgrd = file_utils.read_img(self.background_files[index])
        fx = 1
        fy = 1
        # Resize background to be enough for img
        if w(img) > w(bckgrd):
            fx = w(img) / w(bckgrd)
        if h(img) > h(bckgrd):
            fy = h(img) / h(bckgrd)
        if fx != 1 or fy != 1:
            bckgrd = cv2.resize(bckgrd, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
        max_x = w(bckgrd) - w(img)
        max_y = h(bckgrd) - h(img)

        x = np.random.random_integers(0, max_x)
        y = np.random.random_integers(0, max_y)

        bckgrd_cropped = sub_image(bckgrd, x, y, w(img), h(img))

        paste(bckgrd_cropped, img, 0, 0)

        return bckgrd_cropped
