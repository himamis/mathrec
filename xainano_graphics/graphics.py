from .utils import *


class Graphics:

    def __init__(self, image_loader, post_processor):
        self.image_loader = image_loader
        self.post_processor = post_processor
        self.images = []

    def expression(self, expression):
        image = self.image_loader.image(expression)
        self.images.append(image)

    def fraction(self, numerator_graphics, denominator_graphics):
        numerator = numerator_graphics.draw()
        denominator = denominator_graphics.draw()
        fraction_line = self.image_loader.image("-")
        width = max(w(numerator), w(denominator))
        # TODO: add randomness in width
        padding = 10
        fraction_line = resize(fraction_line, width - padding, h(fraction_line))

        fraction = new_image(width, h(numerator) + h(denominator) + h(fraction_line))
        offset = 0
        paste(fraction, numerator, round((width - w(numerator)) / 2), offset)
        offset += h(numerator)
        paste(fraction, fraction_line, round(padding / 2), offset)
        offset += h(fraction_line)
        paste(fraction, denominator, round((width - w(denominator)) / 2), offset)

        self.images.append(fraction)

    def square_root(self, square_root, expression):
        stretch_start = 20
        stretch_end = 30
        square_root = resize(square_root, w(square_root), h(expression) + 40)
        pre = sub_image(square_root, 0, 0, stretch_start, h(square_root))
        center = sub_image(square_root, stretch_start, 0, stretch_end - stretch_start, h(square_root))
        post = sub_image(square_root, stretch_end, 0, w(square_root), h(square_root))
        center = resize(center, w(expression), h(square_root))
        new_square_root = concat([pre, center, post])
        paste(new_square_root, expression, stretch_start, 30)

        self.images.append(new_square_root)

    def power(self, power):
        power_image = power.draw()
        small_power = resize(power_image, round(w(power_image) / 2), round(h(power_image) / 2))
        padded = pad_image(small_power, 0, 0, round(h(power_image) / 2) + 30, 0)
        # TODO vary the positions

        self.images.append(padded)

    def draw(self):
        return concat(self.images)
