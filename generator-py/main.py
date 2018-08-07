import generators as g
import numpy as np
import string
import cv2
import parser as p
from functools import reduce
import graphics as gr

poly = g.PolynomialGenerator()
expr = g.ExpressionGenerator()
rel = g.RelationGenerator()
var = g.VariableGenerator()
num = g.NumberGenerator()
tok = g.TokenGenerator()


def normal_len(length_mean, length_sd, min=1):
    length = np.random.normal(length_mean, length_sd)
    length = round(abs(length))
    return max(length, min)


def random_polynomial():
    poly.length = np.random.choice([2, 3, 4])
    poly.var_gen.variable_name = np.random.choice(["x", "y", "a", "b"])

    return poly


def random_simple_equation():
    num.separator = np.random.choice([".", ","])

    var.variable_name = np.random.choice(["x", "y"])
    var.multiplier = np.random.choice([None, "\\times"])
    var.scale_generator = np.random.choice([None, num])

    x = [var, num]
    np.random.shuffle(x)
    expr.generators = x
    expr.operators = [np.random.choice([None, "+", "-"]), np.random.choice(["+", "-"])]

    rel.generators = [expr, num]
    rel.operator = np.random.choice(["=", "<", ">", "\\leq", "\\geq"], p=[0.8, 0.05, 0.05, 0.05, 0.05])

    return rel


def coord():
    num = g.NumberGenerator()
    num.p_neg = 0.5
    co = g.function_generator(None, [num, num])

    l = len(string.ascii_uppercase)
    i = np.random.random_integers(0, l - 1)
    tok.token = string.ascii_uppercase[i]
    rel.operator = "="
    rel.generators = [tok, co]

    return rel


def frac_exp():
    pownum = g.NumberGenerator(0, p_real=0)
    pownum.p_neg = 0.4
    num = g.NumberGenerator()
    num.p_neg = 0.5

    pow = g.PowerGenerator(power_generator=pownum)

    var.variable_name = "x"
    var.multiplier = np.random.choice([None, "\\times"])
    var.scale_generator = num
    var.variable_wrapper = pow
    frac = g.fraction_generator(var, pownum)

    e = g.TokenGenerator("e")
    powe = g.PowerGenerator(e, pownum)

    tok.token = "y"
    rel.operator = "="

    rel.generators = [tok, np.random.choice([frac, powe])]
    return rel


def simple_expr_x():
    num = g.NumberGenerator()
    num.separator = np.random.choice([".", ","])

    var = g.VariableGenerator()
    var.variable_name = np.random.choice(["x", "y"])
    var.multiplier = np.random.choice([None, "\\times"])
    var.scale_generator = np.random.choice([None, num])

    x = [var, num]
    np.random.shuffle(x)
    expr = g.ExpressionGenerator()
    expr.generators = x
    expr.operators = [np.random.choice([None]), np.random.choice(["+", "-"])]

    return expr


def long_expr_no_frac():
    cmds = ["sin", "cos", "tan", "cot"]
    cmd_gen = g.function_generator(np.random.choice(cmds), [simple_expr_x()])

    s = simple_expr_x()

    gen = np.random.choice([cmd_gen, s])
    return gen



def long_expr_item():
    frac = g.fraction_generator(long_expr_no_frac(), long_expr_no_frac())
    item = long_expr_no_frac()

    return np.random.choice([frac, item])


def long_expr():
    length = np.random.choice([4, 5, 6, 7])
    expr.generators = [long_expr_item() for _ in range(length)]
    expr.operators = [None]+[np.random.choice(["+", "-"]) for _ in range(length-1)]

    return expr


if __name__ == '__main__':
    tokens = []
    gen = random_simple_equation()
    gen.generate_formula(tokens)

    parser = p.Parser()
    image = parser.parse(tokens)

    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # tokens = []

    # gen = long_expr()
    # gen.generate_formula(tokens)
    # s = reduce((lambda a, b: a + " " + b), tokens)



    # print(s)
