from . import images, graphics, preprocessor

from .images import Images
from .graphics import Graphics


def create_graphics_factory(base):
    pre = preprocessor.Preprocessor()
    image = images.Images(base, pre)

    def create_grahpics(post_processor=None):
        return graphics.Graphics(image, post_processor)

    return create_grahpics
