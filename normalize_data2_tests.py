from normalize_data2 import *

s = ['\\sqrt']
f = ['\\frac']


def test_skip_brackets():
    assert skip_brackets(['1', '{', '2', '}'], 1) == 4
    error = None
    try:
        skip_brackets(['1'], 0)
    except AssertionError as e:
        error = e
    assert error is not None


def test_wrap():
    assert wrap(['1', '2'], 0) == ['{', '1', '}', '2']
    assert wrap(['1', '2'], 1) == ['1', '{', '2', '}']


def test_normalize_under_superscript():
    assert normalize_under_superscript(list("x_1")) == list("x_{1}")
    assert normalize_under_superscript(list("x_{1}")) == list("x_{1}")
    assert normalize_under_superscript(list("x_1^2")) == list("x_{1}^{2}")
    assert normalize_under_superscript(['x', '^', '\\frac', '{', '1', '}', '{', '2', '}']) == \
           ['x', '^', '{', '\\frac', '{', '1', '}', '{', '2', '}', '}']


def test_remove_brackets():
    assert remove_brackets(list("}{{alma(}a}"), 1) == list("}{alma(}a")


def test_normalize_fractions():
    assert normalize_fractions(['\\frac'] + list("x1")) == ['\\frac'] + list("{x}{1}")
    assert normalize_fractions(['\\frac'] + list("{x}1")) == ['\\frac'] + list("{x}{1}")
    assert normalize_fractions(['\\frac'] + list("x{1}")) == ['\\frac'] + list("{x}{1}")
    assert normalize_fractions(['\\frac'] + list("{x}{1}")) == ['\\frac'] + list("{x}{1}")


def test_remove_unnecessary_brackets():
    assert remove_unnecessary_brackets(list("x{1,2,3}{x_{1}}_{{1}}")) == list("x1,2,3x_{1}_{1}")
    assert remove_unnecessary_brackets(list("2x{(9x+1)}{(3x+1)}^{3}")) == list("2x(9x+1)(3x+1)^{3}")
    assert remove_unnecessary_brackets(list("x^{2}{3}")) == list("x^{2}3")
    assert remove_unnecessary_brackets(f + list("{2}{3}")) == f + list("{2}{3}")


def test_normalize_sqrt():
    assert normalize_sqrt(s + list("{almx}") + s + list("x1") + s + list("[3]{x}1") + s + list("[2]xalma")) == \
           s + list("{almx}") + s + list("{x}1") + s + list("[3]{x}1") + s + list("[2]{x}alma")


def test_normalize_under_superscript_order():
    assert normalize_under_superscript_order(list("1+x_{1}^{2}3")) == list("1+x^{2}_{1}3")
    assert normalize_under_superscript_order(list("2+x^{1}_{2}2")) == list("2+x^{1}_{2}2")
    assert normalize_under_superscript_order(list("x^{1}_{2}")) == list("x^{1}_{2}")
    assert normalize_under_superscript_order(list("x_{1}^{2}")) == list("x^{2}_{1}")
    assert normalize_under_superscript_order(list("x^{1}22")) == list("x^{1}22")
    assert normalize_under_superscript_order(list("x_{1}33")) == list("x_{1}33")
    assert normalize_under_superscript_order(list("x_{1111}^{2}3")) == list("x^{2}_{1111}3")


def test_normalize_square_brackets():
    assert normalize_square_brackets(list("[[S]]"), []) == (['\\[', '\\[', 'S', '\\]', '\\]'], [])


def main():
    test_skip_brackets()
    test_wrap()
    test_normalize_under_superscript()
    test_remove_brackets()
    test_normalize_fractions()
    test_remove_unnecessary_brackets()
    test_normalize_sqrt()
    test_normalize_under_superscript_order()
    test_normalize_square_brackets()


if __name__ == "__main__":
    main()
