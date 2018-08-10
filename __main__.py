import cv2
from functools import reduce

from generator import *
from token_parser import Parser
from xainano_graphics import create_graphics_factory

base = "/Users/balazs/university/extracted_images"

config = Config(np.random.choice([",", "."]), c(["x"]), np.random.choice([None, "\\times"]))
tokens = []
gen = random_long_expression()
gen.generate_formula(tokens, config)

token_parser = Parser(create_graphics_factory(base))
image = token_parser.parse(tokens)
s = reduce((lambda a, b: a + " " + b), tokens)
print(s)
print(gen.vocabulary(config))

cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
