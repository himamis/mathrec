import generators as gen
import distributions as d

integer = gen.NumberGenerator(1.0, 1, 2)
fraction = gen.CommandGenerator("\\frac", 2)

random_function_distribution = d.Distribution()
random_function_distribution.set_uniform(["sin", "cos", "tan", "cot", "f", "g", "h"])
random_function = gen.RandomCallableGenerator(random_function_distribution, gen.round_brackets, 1)

operations = d.Distribution()
operations.set_uniform(["+", "-", "*", "/"])
short_operation = gen.ExpressionGenerator(operations, 3, 1)

relation_operations = d.Distribution()
relation_operations.set(["=", "<", ">", "\leq", "\geq"], [0.8, 0.05, 0.05, 0.05, 0.05])
relation = gen.ExpressionGenerator(relation_operations, 2, 0)

