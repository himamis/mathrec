import generators as g
import numpy as np
import cv2
import parser as p
import images as i
import preprocessor as pre
import postprocessor as post
from functools import reduce
from collection import collection


base = "/Users/balazs/university/extracted_images"


if __name__ == '__main__':
    config = g.Config(np.random.choice([",", "."]), collection(["x"]), np.random.choice([None, "\\times"]))
    tokens = []
    gen = g.random_long_expression()
    gen.generate_formula(tokens, config)

    images = i.Images(base, pre.Preprocessor())
    parser = p.Parser(images)
    image = parser.parse(tokens)
    post = post.Postprocessor()
    image = post.postprocess(image)
    s = reduce((lambda a, b: a + " " + b), tokens)
    print(s)
    print(gen.vocabulary(config))

    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
