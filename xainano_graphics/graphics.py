from graphics.utils import *
from graphics import DefaultGraphics, Drawable

kernel = np.ones((3, 3), np.uint8)


class Graphics(DefaultGraphics):

    def __init__(self, image_loader, post_processor):
        super().__init__()
        self.image_loader = image_loader
        self.post_processor = post_processor

    def token(self, token: str):
        image = self.image_loader.image(token)
        xs, ys = np.random.uniform(0.85, 1.15, 2)
        image = resize(image, int(w(image) * xs + 0.5), int(h(image) * ys + 0.5))
        self._concatenator.append(image)

    def fraction(self, numerator: Drawable, denominator: Drawable):
        numerator = numerator.draw()
        denominator = denominator.draw()
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

        self._concatenator.append(fraction, y_center + np.random.randint(-10, 10))

    def square_root(self, expression: Drawable):
        square_root = self.image_loader.image("sqrt")
        expression = expression.draw()
        stretch_start = 20
        stretch_end = 30
        top = 20
        pre = sub_image(square_root, 0, 0, stretch_start, h(square_root))
        cntr = sub_image(square_root, stretch_start, 0, stretch_end - stretch_start, h(square_root))
        post = sub_image(square_root, stretch_end, 0, w(square_root), h(square_root))
        cntr = resize(cntr, w(expression), h(square_root))
        new_square_root = concat([pre, cntr, post])

        image = new_image(w(new_square_root), h(expression) + top)
        paste(image, new_square_root, 0, 0)
        paste(image, expression, stretch_start, top)

        self._concatenator.append(image, top + round(h(expression) / 2))

    def power(self, power: Drawable):
        power_image = power.draw()
        small_power = resize(power_image, round(w(power_image) / 2), round(h(power_image) / 2))

        self._concatenator.append(small_power, h(small_power) + np.random.randint(0, 25))

    def draw(self):
        image = super().draw()
        if self.post_processor is not None:
            image = self.post_processor.postprocess(image)
        return image
