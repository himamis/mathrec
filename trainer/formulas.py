
# supported latex tokens
tokens_numbers = "0123456789"
tokens_alphabet = "abcdefghijklmnopqrstuvwxyz"
tokens_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
tokens_gr_alphabet = ["\\alpha", "\\beta", "\\gamma", "\\delta", "\\epsilon", "\\zeta", "\\eta", "\\theta", "\\iota",
                      "\\kappa", "\\lambda", "\\mu", "\\nu", "\\xi", "\\pi", "\\rho", "\\sigma", "\\tau", "\\upsilon",
                      "\\phi", "\\chi", "\\psi", "\\omega"]
tokens_gr_ALPHABET = ["\\Gamma", "\\Delta", "\\Theta", "\\Lambda", "\\Xi", "\\Upsilon", "\\Phi", "\\Psi", "\\Omega"]
tokens_operations = ["+", "-", "\\cdot", "\\frac", ':', '/', '\\div', '', '', '', '', '', '', '']
tokens_equations = ["=", "<", ">", "\\leq", "\\geq"]
tokens_delimiters = ["|", "|", "(", ")", "[", "]", "\\{", "\\}"]
tokens_functions = ["\\sum", "\\prod", "\\int"]
tokens_others = ["\\prime", "!", "_", "^", ",", ".", "\\%", "\\infty", "{", "}", "\\sqrt"]

MIN_LEN_EQUATION = 2
MAX_LEN_EQUATION = 12

PROB_ORDER_0_SQRT = 0.025
PROB_ORDER_0_SUM = 0.025
PROB_ORDER_0_PROD = 0.025
PROB_ORDER_0_INT = 0.025
PROB_ORDER_0_NUM = 0.05
PROB_ORDER_0_VAR = 0.775
PROB_ORDER_0_FUNC = 0.05
PROB_ORDER_0_PERM = 0.025

PROB_ORDER_N_SQRT = 0.025
PROB_ORDER_N_SUM = 0
PROB_ORDER_N_PROD = 0
PROB_ORDER_N_INT = 0
PROB_ORDER_N_NUM = 0.1
PROB_ORDER_N_VAR = 0.8
PROB_ORDER_N_FUNC = 0.05
PROB_ORDER_N_PERM = 0.025

PROB_PERM_VAR = 0.5

PROB_DELIMS = 0.1
PROB_DELIMS_PARS = 0.2
PROB_DELIMS_BRACKS = 0.4
PROB_DELIMS_ABS = 0.4

PROB_LIMITS_INT_INFTY = 0.2
PROB_LIMITS_SUM_PROD_INFTY = 0.333
PROB_LIMITS_SUM_PROD_NUM = 0.333
PROB_LIMITS_SUM_PROD_VAR = 0.333
PROB_LIMITS_SUM_PROD = 0.25

PROB_FUNC_PARAMS = 0.777
PROB_FUNC_DERIVS = 0.111
PROB_FUNC_PARAMS_DERIVS = 0.111

PROB_EXP_IND = 0.05

PROB_PERCENT = 0.1

PROB_NUM_FLOAT = 0.5
PROB_NUM_INT = 0.5


def create(number):
    formuals = []
    for _ in range(number):
        splits = equation().split()
        equ = ' '.join(splits)
        formulas.append(equ)
    return formulas


def equation():
    length = random.randint(MIN_LEN_EQUATION, MAX_LEN_EQUATION)
    rand = random.randint(0, len(tokens_equations))
    if rand == len(tokens_equations):
        return expression(0, 0, length)
    len_l = length - random.randint(1, length - 1)
    return expression(0, 0, len_l) + ' ' + tokens_equations[rand] + ' ' + expression(0, 0, length - len_l)


def expression():
    rand = random.random()
    if length == 1:
        if order == 0:
            if rand < PROB_ORDER_0_SQRT:
                return sqrt_expression(order, depth, 1)
            elif rand < PROB_ORDER_0_SQRT + PROB_ORDER_0_SUM:
                return sum_expression(order, depth, 1)
            elif rand < PROB_ORDER_0_SQRT + PROB_ORDER_0_SUM + PROB_ORDER_0_PROD:
                return prod_expression(order, depth, 1)
            elif rand < PROB_ORDER_0_SQRT + PROB_ORDER_0_SUM + PROB_ORDER_0_PROD + PROB_ORDER_0_INT:
                return int_expression(order, depth, 1)
            elif rand < PROB_ORDER_0_SQRT + PROB_ORDER_0_SUM + PROB_ORDER_0_PROD + PROB_ORDER_0_INT + PROB_ORDER_0_NUM:
                return number(order, depth)
            elif rand < PROB_ORDER_0_SQRT + PROB_ORDER_0_SUM + PROB_ORDER_0_PROD + PROB_ORDER_0_INT + PROB_ORDER_0_NUM + PROB_ORDER_0_VAR:
                return variable(order, depth)
            elif rand < PROB_ORDER_0_SQRT + PROB_ORDER_0_SUM + PROB_ORDER_0_PROD + PROB_ORDER_0_INT + PROB_ORDER_0_NUM + PROB_ORDER_0_VAR + PROB_ORDER_0_FUNC:
                return func(order, depth)
            else:
                if random.random() < PROB_PERM_VAR:
                    return variable(100, depth) + ' !'
                return int_nr() + ' !'
        else:
            if rand < PROB_ORDER_N_SQRT:
                return sqrt_expression(order, depth, 1)
            elif rand < PROB_ORDER_N_SQRT + PROB_ORDER_N_SUM:
                return sum_expression(order, depth, 1)
            elif rand < PROB_ORDER_N_SQRT + PROB_ORDER_N_SUM + PROB_ORDER_N_PROD:
                return prod_expression(order, depth, 1)
            elif rand < PROB_ORDER_N_SQRT + PROB_ORDER_N_SUM + PROB_ORDER_N_PROD + PROB_ORDER_N_INT:
                return int_expression(order, depth, 1)
            elif rand < PROB_ORDER_N_SQRT + PROB_ORDER_N_SUM + PROB_ORDER_N_PROD + PROB_ORDER_N_INT + PROB_ORDER_N_NUM:
                return number(order, depth)
            elif rand < PROB_ORDER_N_SQRT + PROB_ORDER_N_SUM + PROB_ORDER_N_PROD + PROB_ORDER_N_INT + PROB_ORDER_N_NUM + PROB_ORDER_N_VAR:
                return variable(order, depth)
            elif rand < PROB_ORDER_N_SQRT + PROB_ORDER_N_SUM + PROB_ORDER_N_PROD + PROB_ORDER_N_INT + PROB_ORDER_N_NUM + PROB_ORDER_N_VAR + PROB_ORDER_N_FUNC:
                return func(order, depth)
            else:
                if random.random() < PROB_PERM_VAR:
                    return variable(100, depth) + ' !'
                return int_nr() + ' !'
    len_l, len_r = split_length(length)
    expr_l = expression(order, depth + 1, len_l)
    expr_r = expression(order, depth + 1, len_l)
    operator = random.choice(tokens_operations)
    if operator == '\\frac':
        return operator + ' { ' + expr_l + ' } { ' + expr_r + ' }'
    if random.random() < PROB_DELIMS:
        rand = random.random()
        if rand < PROB_DELIMS_PARS:
            delims = ['( ', ' )']
        elif rand < PROB_DELIMS_PARS + PROB_DELIMS_BRACKS:
            delims = ['[ ', ' ]']
        else:
            delims = ['| ', ' |']
    else:
        delims = ['', '']
    return delims[0] + expr_l + ' ' + operator + ' ' + expr_r + delims[1]


def split_length(length):
    len1 = int(length/2)
    return len1, length - len1


def sqrt_expression(order, depth, length):
    expo = exponent(order, depth)[5:-1]
    if expo == '':
        return '\\sqrt { ' + expression(order, depth + 1, length) + ' }'
    else:
        return '\\sqrt [ ' + expo + " ] { " + expression(order, depth + 1, length) + ' }'


def int_expression(order, depth, length):
    s = '\\int '
    expo = exponent(order, depth)
    if expo != '' and random.random() < PROB_LIMITS_INT_INFTY:
        expo = ' ^ { \\infty }'
    s += expo
    ind = index(order, depth)
    if ind != '' and random.random() < PROB_LIMITS_INT_INFTY:
        ind = ' _ { - \\infty }'
    s += ind
    return s + ' ' + expression(order, depth + 1, length) + ' d ' + variable(1000, depth)


def sum_expression(order, depth, length):
    s = '\\sum '
    s += limits()
    return s + ' ' + expression(order, depth + 1, length)


def prod_expression(order, depth, length):
    s = '\\prod '
    s += limits()
    return s + ' ' + expression(order, depth + 1, length)


def limits():
    s = ''
    if random.random() < PROB_LIMITS_SUM_PROD:
        rand = random.random()
        if rand < PROB_LIMITS_SUM_PROD_INFTY:
            s += ' ^ {  \\infty }'
        elif rand < PROB_LIMITS_SUM_PROD_INFTY + PROB_LIMITS_SUM_PROD_NUM:
            s += ' ^ { ' + int_nr() + ' }'
        else:
            s += ' ^ { ' + variable(1000, 0) + ' }'

    if random.random() < PROB_LIMITS_SUM_PROD:
        s += ' _ { ' + variable(1000, 0) + ' = ' + int_nr() + ' }'

    return s


def func(order, depth):
    foo = variable(1000, depth)
    nr_var = random.randint(0, 3)
    nr_deris = random.randint(1, 3)
    var = ' ( '
    for _ in range(nr_var):
        if random.random() < 0.9:
            var += variable(1000, depth)
        else:
            var += number(1000, depth)
        var += ' , '
    if var[-2] == ',':
        var = var[:-2] + ' )'
    else:
        var += ' )'
    deris = ''
    for _ in range(nr_deris):
        deris += ' \\prime'
    rand = random.random()
    if rand < PROB_FUNC_PARAMS:
        return foo + var
    elif rand < PROB_FUNC_PARAMS + PROB_FUNC_PARAMS_DERIVS:
        return foo + var + deris
    else:
        return foo + deris

    
def variable(order, depth):
    rand = random.randint(0, len(tokens_alphabet) + len(tokens_ALPHABET) + len(tokens_gr_alphabet) +
                          len(tokens_gr_ALPHABET) - 1)
    if rand < len(tokens_alphabet):
        var = tokens_alphabet[rand]
    elif rand < len(tokens_alphabet) + len(tokens_ALPHABET):
        var = tokens_ALPHABET[rand - 26]
    elif rand < len(tokens_alphabet) + len(tokens_ALPHABET) + len(tokens_gr_alphabet):
        var = tokens_gr_alphabet[rand - 52]
    else:
        var = tokens_gr_ALPHABET[rand - 75]

    var += exponent(order, depth)
    var += index(order, depth)
    return var


def exponent(order, depth):
    return exp_ind(order, depth, '^')


def index(order, depth):
    return exp_ind(order, depth, '_')


def exp_ind(order, depth, case):
    max_len = max(0, 3-order)
    length = random.randint(0, max_len)
    if length > 0 and random.random() < PROB_EXP_IND / (order + 1) / (order + 1):
        expr = expression(order + 1, depth, length)
        return ' ' + case + ' { ' + expr + ' }'
    return ""


def number(order, depth):
    if random.random() < PROB_NUM_FLOAT:
        nr =  float_nr()
    else:
        nr = int_nr()
    expo = exponent(order, depth)
    if expo == '' and random.random() < PROB_PERCENT:
        return nr + ' \\%'
    return nr + expo


def float_nr():
    return nr_to_text(random.uniform(0.0, 99.99))


def int_nr():
    return nr_to_text(random.randint(0, 99))


def nr_to_text(nr):
    s = str(nr)
    if len(s) > 5: s = s[0:5]
    st = ""
    for c in s:
        st += c + " "
    return st[:-1]
