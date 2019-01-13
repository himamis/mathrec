import numpy as np
import string
from .collection import c, Collection
from typing import List

'''
Generators should be classes that have at least two public methods with the following signatures
generate_formula([String])
vocabulary(config)
The first one should generate a random and valid mathematical expression in latex
broken down into the smallest tokens possible (e.g. sin should be 's', 'i', 'n', instead of 'sin').
The second method should return all tokens that the generator can generate in a set.
'''


class Config:
    """ Configs are used to modify the behaviour of generators."""

    def __init__(self, separator='.', variables=c(['x', 'y']), multiplier=c()):
        self.separator = separator
        self.variables = variables
        self.multiplier = multiplier

    def vocabulary(self):
        return self.variables.all() | self.multiplier.all() | {self.separator}


class FormulaGenerator(object):
    """ FormulaGenerators are objects that can generate formulas."""

    def generate(self, tokens: List[str], config=Config()):
        """
        Generates a formula.

        :param tokens: a list that contains tokens
        :param config: optional config object
        """
        raise NotImplementedError("Class %s doesn't implement generate()" % self.__class__.__name__)

    def vocabulary(self, config=Config()):
        """
        Creates a vocabulary that contains all tokens that this generator can create.

        :param config: optional config
        :return: set of tokens
        """
        raise NotImplementedError("Class %s doesn't implement vocabulary()" % self.__class__.__name__)


class WrappingFormulaGenerator(FormulaGenerator):
    """ Formula generators that wrap other generators.
        It can accept arbitrary generators in a list of generators.
    """

    def __init__(self, generators: List[FormulaGenerator]):
        super().__init__()
        self.generators = generators

    def generate(self, tokens, config=Config()):
        super().generate(tokens, config)

    def vocabulary(self, config=Config()):
        vocab = set()
        for generator in self.generators:
            vocab = vocab | generator.vocabulary(config)
        return vocab


class RandomTokenGenerator(FormulaGenerator):
    """ Generates a random token """

    def __init__(self, token_collection: Collection):
        super().__init__()
        self.token_collection = token_collection

    def generate(self, tokens: List[str], config=Config()):
        tokens += [self.token_collection.get()]

    def vocabulary(self, config=Config()):
        return self.token_collection.all()


class TokenGenerator(RandomTokenGenerator):
    """ Generates a single token. """

    def __init__(self, token="x"):
        super().__init__(c(token))

    @property
    def token(self):
        return self.token_collection.get()

    @token.setter
    def token(self, value):
        self.token_collection = c(value)


class NumberGenerator(FormulaGenerator):
    """ Generates numbers. """

    def __init__(self, n=5, p=0.05, p_real=0.1, p_neg=0.0):
        """ Create a NumberGenerator. The length of the number (number of decimals)
        is defined by a binomial distribution with parameters n and p.
        With probability p_real, it generates a real number.
        With probability p_neg it generates a negative number.

        :param n: binomial probability distribution parameter n
        :param p: binomial probability distribution parameter p
        :param p_real: probability of real number
        :param p_neg: probability of negative number
        """
        self.n = n
        self.p = p
        self.p_real = p_real
        self.p_neg = p_neg

    def generate(self, tokens, config=Config()):
        if np.random.random() < self.p_neg:
            tokens += "-"
        self._generate_number(tokens)
        if np.random.random() < self.p_real:
            tokens += config.separator
            self._generate_number(tokens)

    def _generate_number(self, tokens):
        length = np.random.binomial(self.n, self.p) + 1
        if length > 0:
            tokens += [str(np.random.random_integers(1, 9))]
            tokens += [str(np.random.random_integers(0, 9)) for _ in range(length - 1)]

    def vocabulary(self, config=Config()):
        vocab = {str(num) for num in range(0, 10)}
        if self.p_neg > 0:
            vocab = vocab | {'-'}
        if self.p_real > 0:
            vocab = vocab | {config.separator}
        return vocab


class ExpressionGenerator(WrappingFormulaGenerator):
    """ Generates expressions. """

    def __init__(self, generators: List[FormulaGenerator], operators: List[Collection] = list()):
        """
        Creates an expression generator.
        It is defined by a list of generators, which are separated by the operators.

        :param generators: the list of generators
        :param operators: operators that separate the expression. Each operator is a collection of operators.
        """
        super().__init__(generators)
        self.operators = operators
        self.randomize_order = False

    def generate(self, tokens, config=Config()):
        if self.randomize_order:
            np.random.shuffle(self.generators)

        for index in range(len(self.generators)):
            if index < len(self.operators) and self.operators[index] is not None:
                operator = self.operators[index].get()
                if operator is not None:
                    tokens += [operator]
            self.generators[index].generate(tokens, config)

    def vocabulary(self, config=Config()):
        vocab = set()
        for generator in self.generators:
            vocab = vocab | generator.vocabulary(config)
        for operator in self.operators:
            if operator is not None:
                vocab = vocab | operator.all()
        if None in vocab:
            vocab.remove(None)
        return vocab


class CommandGenerator(WrappingFormulaGenerator):
    """ Generates formulas with commands and parameters. """

    def __init__(self, name, generators=list()):
        """ Create a CommandGenerator.

        :param name: name of the command
        :param generators: parameter generators, separated by curly braces {}
        """
        super().__init__(generators)
        self.name = name

    def generate(self, tokens, config=Config()):
        tokens += [self.name]
        for generator in self.generators:
            tokens += "{"
            generator.generate(tokens, config)
            tokens += "}"

    def vocabulary(self, config=Config()):
        vocab = {self.name}
        for generator in self.generators:
            vocab = vocab | generator.vocabulary(config)
        return vocab


class CallableGenerator(WrappingFormulaGenerator):
    """ Callable generator e.g. sin(x) or f(a,b). """

    def __init__(self, name, generators, brackets):
        """ Creates a CallableGenerator.

        :param name: name of the generator
        :param generators: parameters
        :param brackets: brackets to use
        """
        super().__init__(generators)
        self._name = c(name)
        self.brackets = brackets

    @property
    def name(self):
        """ Get the name of the callable e.g. sin """
        return self._name

    @name.setter
    def name(self, value):
        """ Set the name of the callable e.g. cos """
        self._name = c(value)

    def generate(self, tokens, config=Config()):
        if self.name is not None:
            name = self.name.get()
            if name is not None:
                tokens.append(name)

        tokens += [self.brackets[0]]

        for index in range(len(self.generators)):
            self.generators[index].generate(tokens, config)
            if index < len(self.generators) - 1:
                tokens += [","]
        tokens += [self.brackets[1]]

    def vocabulary(self, config=Config()):
        vocab = {self.brackets[0], self.brackets[1]}
        if self.name is not None:
            for name in self.name.all():
                if name is not None:
                    vocab.add(name)
        for generator in self.generators:
            vocab = vocab | generator.vocabulary(config)
        if len(self.generators) > 1:
            vocab = vocab | {","}
        if None in vocab:
            vocab.remove(None)
        return vocab


class RelationGenerator(ExpressionGenerator):
    """ Relation generator with two left and right sides. """

    def __init__(self, operator=c("="),
                 left_side: FormulaGenerator = TokenGenerator(),
                 right_side: FormulaGenerator = TokenGenerator()):
        super().__init__([left_side, right_side], [operator])


class VariableGenerator(FormulaGenerator):
    """ Generates variables. """

    def __init__(self, variable_wrapper: WrappingFormulaGenerator = None, scale_generator: FormulaGenerator = None):
        """ Create a VariableGenerator.

        :param variable_wrapper:
        :param scale_generator: token generator that is appended before the variable
        """
        self.variable_wrapper = variable_wrapper
        self.scale_generator = scale_generator

    def generate(self, tokens, config=Config()):
        if self.scale_generator is not None:
            self.scale_generator.generate(tokens, config)
            if config.multiplier is not None:
                multiplier = config.multiplier.get()
                if multiplier is not None:
                    tokens += [multiplier]
        if self.variable_wrapper is not None:
            self.variable_wrapper.generators = [TokenGenerator(config.variables.get())]
            self.variable_wrapper.generate(tokens, config)
        else:
            tokens += [config.variables.get()]

    def vocabulary(self, config=Config()):
        vocab = set()
        if self.scale_generator is not None:
            vocab = vocab | self.scale_generator.vocabulary(config)
            if config.multiplier is not None:
                vocab = vocab | config.multiplier.all()
        if self.variable_wrapper is not None:
            vocab = vocab | self.variable_wrapper.vocabulary(config)
            vocab = vocab | config.variables.all()
        return vocab


class PowerGenerator(WrappingFormulaGenerator):
    """ Generates power expressions. """

    def __init__(self, generator: FormulaGenerator = NumberGenerator(),
                 power_generator: FormulaGenerator = TokenGenerator("2")):
        """ Create a PowerGenerator object.

        :param generator: object to wrap with power
        :param power_generator: the expression in the power term
        """
        super().__init__([generator])
        self.power_generator = power_generator

    def generate_formula(self, tokens, config):
        new_tokens = []
        self.generators[0].generate(new_tokens, config)
        if len(new_tokens) > 1:
            tokens += "("
        tokens += new_tokens
        if len(new_tokens) > 1:
            tokens += ")"
        tokens += ["^", "{"]
        self.power_generator.generate(tokens, config)
        tokens += ["}"]

    def vocabulary(self, config=Config()):
        vocab = {"(", ")", "^", "{", "}"}
        vocab = vocab | self.generators[0].vocabulary(config)
        vocab = vocab | self.power_generator.vocabulary(config)
        return vocab


# These build on the ones before
class PolynomialGenerator(FormulaGenerator):
    """ Generates polynomial expressions. """

    def __init__(self, length=c([2, 3, 4]), p_miss=0.1, p_minus=0.3):
        """ Creates a Polynomial Generator object.

        :param length: collection of lengths
        :param p_miss: probability of missing a term
        :param p_minus: probability of having a negative term
        """
        self.length = length
        self.p_miss = p_miss
        self.p_minus = p_minus

        self.power = TokenGenerator("1")
        self.power_gen = PowerGenerator(TokenGenerator(), power_generator=self.power)

        self.number_gen = NumberGenerator()
        self.var_gen = VariableGenerator(self.power_gen, self.number_gen)

    def generate(self, tokens: List[str], config=Config()):
        new_tokens = []
        length = self.length.get()
        for ind in range(length, 0, -1):
            if not (ind == 2 and length == 2) and np.random.random() < self.p_miss:
                continue
            if np.random.random() < self.p_minus:
                new_tokens += ["-"]
            elif len(new_tokens) > 0:
                new_tokens += ["+"]

            if ind > 2:
                self.power.token = str(ind - 1)
                self.var_gen.variable_wrapper = self.power_gen
                self.var_gen.generate(new_tokens, config)
            elif ind == 2:
                self.var_gen.variable_wrapper = None
                self.var_gen.generate(new_tokens, config)
            else:
                self.number_gen.generate(new_tokens, config)
        tokens += new_tokens

    def vocabulary(self, config=Config()):
        vocab = {"+"} | self.var_gen.vocabulary(config)
        if self.p_minus > 0:
            vocab = vocab | {"-"}
        return vocab


class RandomGenerator(FormulaGenerator):
    """ A collection of generators. """

    def __init__(self, generators: List[FormulaGenerator]):
        """ Create a RandomGenerator.

        :param generators: a list of generators
        """
        self.generators = generators

    def generate(self, tokens: List[str], config=Config()):
        generator = np.random.choice(self.generators)
        generator.generate(tokens, config)

    def vocabulary(self, config=Config()):
        vocab = set()
        for generator in self.generators:
            vocab = vocab | generator.vocabulary(config)
        return vocab


class GibberishGenerator(WrappingFormulaGenerator):
    """ Generates just unintelligible gibberish. """

    def __init__(self, generators, min_length=4, max_length=20, brckt_p=0.2):
        super().__init__(generators)
        self.min_length = min_length
        self.max_length = max_length
        self.brckt_p = brckt_p
        self.brackets = [('(', ')'), ('[', ']'), ('{', '}')]

    def generate(self, tokens: List[str], config=Config()):
        length = np.random.randint(self.min_length, self.max_length)
        if length > 3 and np.random.uniform() < self.brckt_p:
            opening_bracket = np.random.randint(0, length - 2)
            closing_bracket = np.random.randint(opening_bracket, length - 1)
        else:
            opening_bracket = -2
            closing_bracket = -2

        bracket_idx = np.random.choice(len(self.brackets))
        bracket = self.brackets[bracket_idx]

        for i in range(0, length):
            if i == opening_bracket:
                tokens += bracket[0]
            elif i == closing_bracket:
                tokens += bracket[1]
            else:
                idx = np.random.choice(len(self.generators))
                generator = self.generators[idx]
                generator.generate(tokens, config)

    def vocabulary(self, config=Config()):
        tokens = super().vocabulary(config)
        for bracket in self.brackets:
            tokens |= {bracket[0], bracket[1]}
        return tokens


def gibberish_generator(min_length=4, max_length=20, brckt_p=0.2):
    random_tokens = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                     '\\Delta', '\\alpha', '\\beta', '\\cos', '\\div', '\\exists', '\\forall', '\\gamma', '\\geq',
                     '\\gt', '\\in', '\\infty', '\\int', '\\lambda', '\\ldots', '\\leq', '\\lim', '\\log', '\\lt',
                     '\\mu', '\\neq', '\\phi', '\\pi', '\\pm', '\\prime', '\\rightarrow', '\\sigma', '\\sin',
                     '\\sqrt', '\\sum', '\\tan', '\\theta', '\\times',
                     '!', '+', ',', '-', '.', '/', '=']
    generator = RandomTokenGenerator(c(random_tokens))
    single_token_gen = RandomGenerator([uppercase_character(), lowercase_character(), generator])
    return GibberishGenerator([single_token_gen], min_length, max_length, brckt_p)


square_brackets = ("[", "]")
round_brackets = ("(", ")")
curly_brackets = ("{", "}")


def numbers():
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    return RandomTokenGenerator(c(numbers))


def uppercase_character():
    characters = ['A', 'B', 'C', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'X', 'Y']
    return RandomTokenGenerator(c(characters))


def lowercase_character():
    characters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                  'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    return RandomTokenGenerator(c(characters))


def function_generator(name, generators): return CallableGenerator(name, generators, round_brackets)


def fraction_generator(numerator, denominator): return CommandGenerator("\\frac", [numerator, denominator])


def random_simple_expression():
    num = NumberGenerator()

    var1 = VariableGenerator(scale_generator=num)
    var2 = VariableGenerator()
    rand = RandomGenerator([var1, var2])

    expr = ExpressionGenerator([rand, num], operators=[c(None), c(["+", "-"])])
    expr.randomize_order = True

    return expr


def random_simple_equation():
    num = NumberGenerator()
    var = VariableGenerator(scale_generator=num)

    expr = ExpressionGenerator([var, num], [c([None, "+", "-"]), c(["+", "-"])])
    expr.randomize_order = True

    rel = ExpressionGenerator([expr, num], [c(None), c(["=", "\\leq", "\\geq"])])

    return rel


def random_coord():
    num = NumberGenerator(p_neg=0.5)
    co = function_generator(None, [num, num])

    generators = []
    for character in string.ascii_uppercase:
        generators.append(TokenGenerator(character))
    rand = RandomGenerator(generators)

    return RelationGenerator(operator=c("="), left_side=rand, right_side=co)


def random_fraction():
    num = NumberGenerator(p_neg=0.5)
    pow_num = NumberGenerator(0, p_real=0, p_neg=0.4)
    power_generator = PowerGenerator(power_generator=pow_num)
    var = VariableGenerator(scale_generator=num, variable_wrapper=power_generator)

    fraction = fraction_generator(var, pow_num)

    tok = TokenGenerator("y")
    rel = ExpressionGenerator(generators=[tok, RandomGenerator([fraction])], operators=[c(None), c("=")])

    return rel


def random_long_expression_no_frac():
    commands = ["sin", "cos", "tan"]
    simple = random_simple_expression()
    command_gen = function_generator(c(commands), [simple])
    random = RandomGenerator([command_gen, simple])

    return random


def _random_long_expr_item():
    expression = random_long_expression_no_frac()
    frac = fraction_generator(expression, expression)

    return RandomGenerator([expression, frac])


def random_long_expression():
    generators = []

    for i in range(4, 8):
        expr = ExpressionGenerator(generators=[_random_long_expr_item() for _ in range(i)],
                                   operators=[c(None)] + [c(["+", "-"]) for _ in range(i - 1)])
        generators.append(expr)

    return RandomGenerator(generators)


def random_square_root():
    simple = random_simple_expression()
    return CommandGenerator("\\sqrt", [simple])


def random_generator():
    generators = [random_simple_expression()]
    # generators = [random_simple_expression(), random_polynomial(), random_coord(),
    #               random_fraction(), random_long_expression_no_frac()]\
    #     , almost_absolutely_random_generator(),
    #              almost_absolutely_random_generator(), almost_absolutely_random_generator()]
    return RandomGenerator(generators)


def almost_absolutely_random_generator():
    long_random_generator = gibberish_generator()
    short_random_generator = gibberish_generator(2, 5)
    very_short_generator = gibberish_generator(1, 2)
    bs_frac_generator = fraction_generator(short_random_generator, short_random_generator)
    random_frac_or_nofrac = RandomGenerator([short_random_generator, bs_frac_generator])
    relation_generator = RelationGenerator(left_side=random_frac_or_nofrac, right_side=random_frac_or_nofrac)
    power_generator = PowerGenerator(short_random_generator, very_short_generator)
    longer_power = ExpressionGenerator(generators=[power_generator, very_short_generator])
    power_relation = RelationGenerator(left_side=random_frac_or_nofrac, right_side=longer_power)
    generators = [long_random_generator, bs_frac_generator, short_random_generator, bs_frac_generator,
                  relation_generator, power_generator, longer_power, power_relation]
    return RandomGenerator(generators)


def single_token_generator():
    generators = []
    for char in range(0, 9):
        generators.append(TokenGenerator(str(char)))
    for operators in ['=', '+', '-', '.', '\\times', '(', ')', ',']:
        generators.append(TokenGenerator(operators))
    for char in range(ord('a'), ord('z') + 1):
        generators.append(TokenGenerator(chr(char)))

    return RandomGenerator(generators)


def simple_number_operation_generator() -> GibberishGenerator:
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '+']
    token = RandomTokenGenerator(c(numbers))
    return GibberishGenerator([token], 1, 6, 0)
