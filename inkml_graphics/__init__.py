from .graphics import InkmlGraphics


def create_graphics_factory(base):
    import pickle
    token_groups = pickle.load(open(base, 'rb'))

    def create_grahpics():
        return InkmlGraphics(token_groups)

    return create_grahpics
