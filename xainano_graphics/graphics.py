from graphics.utils import *
import math

kernel = np.ones((3, 3), np.uint8)


class Graphics:

    def __init__(self, image_loader, post_processor):
        self.image_loader = image_loader
        self.post_processor = post_processor
        self.images = []

    def expression(self, expression):
        image = self.image_loader.image(expression)
        xs, ys = np.random.uniform(0.85, 1.15, 2)
        image = resize(image, int(w(image) * xs + 0.5), int(h(image) * ys + 0.5))
        self._add_image(image)

    def fraction(self, numerator_graphics, denominator_graphics):
        numerator = numerator_graphics.draw()
        denominator = denominator_graphics.draw()
        fraction_line = self.image_loader.image("-")
        width = max(w(numerator), w(denominator))
        fraction_line_width = max(width + np.random.randint(-40, 40), 20)
        fraction_line = resize(fraction_line, fraction_line_width, h(fraction_line))
        width = max(width, fraction_line_width)

        fraction = new_image(width, h(numerator) + h(denominator) + h(fraction_line))
        offset = 0
        # Add variance to insert position in y
        paste(fraction, numerator, round((width - w(numerator)) / 2), offset)
        offset += h(numerator) - np.random.randint(0, 15)

        paste(fraction, fraction_line, round((width - fraction_line_width) / 2), offset)
        offset += h(fraction_line)
        y_center = offset - round(h(fraction_line) / 2)
        paste(fraction, denominator, round((width - w(denominator)) / 2), offset)

        self._add_image(fraction, y_center + np.random.randint(-10, 10))

    def square_root(self, expression_graphics):
        square_root = self.image_loader.image("sqrt")
        expression = expression_graphics.draw()
        stretch_start = 20
        stretch_end = 30
        top = 20
        pre = sub_image(square_root, 0, 0, stretch_start, h(square_root))
        center = sub_image(square_root, stretch_start, 0, stretch_end - stretch_start, h(square_root))
        post = sub_image(square_root, stretch_end, 0, w(square_root), h(square_root))
        center = resize(center, w(expression), h(square_root))
        new_square_root = concat([pre, center, post])


        image = new_image(w(new_square_root), h(expression) + top)
        paste(image, new_square_root, 0, 0)
        paste(image, expression, stretch_start, top)

        self._add_image(image, top + round(h(expression) / 2))

    def power(self, power):
        power_image = power.draw()
        small_power = resize(power_image, round(w(power_image) / 2), round(h(power_image) / 2))

        self._add_image(small_power, h(small_power) + np.random.randint(0, 25))

    def _add_image(self, image, y_center=None):
        if y_center is None:
            y_center = round(h(image) / 2)
        self.images.append((image, y_center))

    def draw(self):
        image = self._concat_images()
        if self.post_processor is not None:
            image = self.post_processor.postprocess(image)
        return image

    def _concat_images(self):
        width = 0
        height = 0
        paddings = []
        for image, y_center in self.images:
            padding = np.random.randint(0, 15)
            paddings.append(padding)
            width += w(image) + padding
            top_missing = abs(min(round(height / 2) - y_center, 0))
            bottom_missing = max(round(height / 2) + (h(image) - y_center) - height, 0)
            missing = max(top_missing, bottom_missing)
            height = height + missing * 2

        concat_image = new_image(width, height)

        offset = 0
        for i, (image, y_center) in enumerate(self.images):
            paste(concat_image, image, offset, round(abs(h(concat_image) / 2 - y_center)))
            offset += w(image) + paddings[i]
        return concat_image