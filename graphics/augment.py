import file_utils
from graphics.utils import *
import math
from imgaug import augmenters as aa

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

    def __init__(self, background_images, grid_images):
        self.background_files = [x for x in file_utils.list_files(background_images) if x.lower().endswith(".jpg") or x.lower().endswith(".jpeg")]
        self.grid_images = [x for x in file_utils.list_files(grid_images) if x.lower().endswith(".jpg") or x.lower().endswith('jpeg')]
        self.grid_percentage = 0.7
        self.shadow_percentage = 0.5
        self.pers = aa.PerspectiveTransform(scale=(0.01, 0.1))
        self.noise = aa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255))

    def augment(self, image):
        image = self._background(image)
        image = self._shadow(image)
        image = self._blur(image)
        image = self._static(image)
        image = self._grayscale(image)
        image = self._perspective(image)
        return image

    def size_changing_augment(self, img):
        return self._rotate(img)

    def _rotate(self, img):
        angle = np.random.random_integers(-5, 5)
        return _rotate_bound(img, angle)

    def _blur(self, img):
        num = np.random.random_integers(0, 2) * 2 + 1
        sigma = np.random.random_integers(1, 5)
        return cv2.GaussianBlur(img, (num, num), sigma)

    def _background(self, img):
        bckgrd = self._random_fittin_image(img, self.background_files)
        if np.random.uniform(0, 1) < self.grid_percentage:
            grid = self._random_fittin_image(img, self.grid_images)
            paste(bckgrd, grid, 0, 0)

        paste(bckgrd, img, 0, 0)

        return bckgrd

    def _random_fittin_image(self, image, images):
        index = np.random.random_integers(0, len(images) - 1)
        bckgrd = file_utils.read_img(images[index])
        fx = 1
        fy = 1
        # Resize background to be enough for img
        if w(image) > w(bckgrd):
            fx = w(image) / w(bckgrd)
        if h(image) > h(bckgrd):
            fy = h(image) / h(bckgrd)
        if fx != 1 or fy != 1:
            bckgrd = cv2.resize(bckgrd, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
        max_x = w(bckgrd) - w(image)
        max_y = h(bckgrd) - h(image)

        x = np.random.random_integers(0, max_x)
        y = np.random.random_integers(0, max_y)

        return sub_image(bckgrd, x, y, w(image), h(image))

    def _shadow(self, image):
        if np.random.uniform(0, 1) < self.shadow_percentage:
            shadow = self._create_shadow_image(w(image), h(image))
            image = cv2.subtract(image, shadow)
        return image

    def _create_shadow_image(self, width, height):
        shadow = np.zeros((height, width), dtype=np.uint8)
        it = np.nditer(shadow, flags=['multi_index'], op_flags=['readwrite'])
        r = np.random.random_integers(min(width/2, height/2), min(width, height))
        a, b = np.random.random_integers(0, max(width, height), 2)
        m = 1000
        shadow_intensity = np.random.random_integers(50, 70)
        while not it.finished:
            x, y = it.multi_index
            val = math.pow((x - a), 2) + math.pow((y - b), 2) - math.pow(r, 2)
            if val < -m:
                it[0] = shadow_intensity
            elif val < m:
                it[0] = (1 - ((val + m) / (2 * m))) * shadow_intensity
            it.iternext()


        shadow = np.repeat(shadow[:, :, np.newaxis], 3, axis=2)

        return shadow

    def _static(self, image):
        return self.noise.augment_image(image)

    def _grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _perspective(self, image):
        return self.pers.augment_image(image)
