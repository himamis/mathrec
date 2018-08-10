from . import images, graphics, preprocessor, utils


def create_graphics_factory(base):
    pre = preprocessor.Preprocessor()
    image = images.Images(base, pre)

    def create_grahpics():
        return graphics.Graphics(image)

    return create_grahpics
