from normalize_data2 import *


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


def test_normalize_sqrt():
    s = ['\\sqrt']
    assert normalize_sqrt(s + list("{almx}") + s + list("x1") + s + list("[3]{x}1") + s + list("[2]xalma")) == \
           s + list("{almx}") + s + list("{x}1") + s + list("[3]{x}1") + s + list("[2]{x}alma")


def main():
    test_skip_brackets()
    test_wrap()
    test_normalize_under_superscript()
    test_remove_brackets()
    test_normalize_fractions()
    test_remove_unnecessary_brackets()
    test_normalize_sqrt()


if __name__ == "__main__":
    main()