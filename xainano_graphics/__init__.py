from . import images, graphics, preprocessor, utils, postprocessor

from .images import Images
from .graphics import Graphics


def create_graphics_factory(base):
    pre = preprocessor.Preprocessor()
    post = postprocessor.Postprocessor()
    image = images.Images(base, pre)

    def create_grahpics():
        return graphics.Graphics(image, post)

    return create_grahpics
