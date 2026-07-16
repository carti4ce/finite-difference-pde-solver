"""Safe-expression-evaluator tests.

The evaluator is the only place user-supplied text is interpreted, so the
security tests here (rejecting anything outside a numeric-expression subset)
matter as much as the numeric ones.
"""
import numpy as np
import pytest

from expr import ExpressionError, evaluate_expression


# ---------------------------------------------------------------------------
# accepted expressions
# ---------------------------------------------------------------------------

def test_basic_arithmetic():
    assert evaluate_expression("2 + 3 * 4", {}) == 14
    assert evaluate_expression("(2 + 3) * 4", {}) == 20
    assert evaluate_expression("2 ** 10", {}) == 1024
    assert evaluate_expression("-5 + 2", {}) == -3
    assert evaluate_expression("7 % 3", {}) == 1


def test_constants():
    assert evaluate_expression("pi", {}) == pytest.approx(np.pi)
    assert evaluate_expression("e", {}) == pytest.approx(np.e)


def test_variables_vectorize():
    x = np.array([0.0, 0.5, 1.0])
    result = evaluate_expression("sin(pi * x)", {"x": x})
    assert np.allclose(result, np.sin(np.pi * x))


def test_two_variables():
    x = np.array([0.2, 0.4])
    y = np.array([0.1, 0.9])
    result = evaluate_expression("x * y + 1", {"x": x, "y": y})
    assert np.allclose(result, x * y + 1)


@pytest.mark.parametrize("fn,arg,expected", [
    ("sin", 0.0, 0.0), ("cos", 0.0, 1.0), ("exp", 0.0, 1.0),
    ("sqrt", 4.0, 2.0), ("abs", -3.0, 3.0), ("tanh", 0.0, 0.0),
])
def test_whitelisted_functions(fn, arg, expected):
    assert evaluate_expression(f"{fn}({arg})", {}) == pytest.approx(expected)


def test_ternary_where():
    x = np.array([-1.0, 0.5, 2.0])
    result = evaluate_expression("1 if x > 1 else 0", {"x": x})
    assert np.array_equal(result, [0, 0, 1])


def test_nonlinear_reaction_style_expression():
    u = np.array([0.0, 0.5, 1.0])
    result = evaluate_expression("u * (1 - u)", {"u": u})
    assert np.allclose(result, u * (1 - u))


# ---------------------------------------------------------------------------
# rejected: unknown names / functions
# ---------------------------------------------------------------------------

def test_unknown_variable_rejected():
    with pytest.raises(ExpressionError):
        evaluate_expression("z + 1", {"x": np.array([1.0])})


def test_unknown_function_rejected():
    with pytest.raises(ExpressionError):
        evaluate_expression("open('f')", {})


def test_empty_expression_rejected():
    with pytest.raises(ExpressionError):
        evaluate_expression("", {})
    with pytest.raises(ExpressionError):
        evaluate_expression("   ", {})


# ---------------------------------------------------------------------------
# rejected: sandbox-escape attempts (the security-critical cases)
# ---------------------------------------------------------------------------

DANGEROUS_EXPRESSIONS = [
    "__import__('os').system('echo pwned')",
    "os.system('echo pwned')",
    "x.__class__",
    "().__class__.__bases__[0].__subclasses__()",
    "(1).__class__.__mro__",
    "open('/etc/passwd').read()",
    "exec('1')",
    "eval('1')",
    "[i for i in range(3)]",
    "{i: i for i in range(3)}",
    "lambda: 1",
    "x if True else (lambda: 1)()",
    "globals()",
    "locals()",
    "__builtins__",
    "x; y",
    "import os",
]


@pytest.mark.parametrize("expression", DANGEROUS_EXPRESSIONS)
def test_dangerous_expressions_rejected(expression):
    with pytest.raises(ExpressionError):
        evaluate_expression(expression, {"x": np.array([1.0]), "y": np.array([2.0])})


def test_invalid_syntax_rejected():
    with pytest.raises(ExpressionError):
        evaluate_expression("sin(", {})
    with pytest.raises(ExpressionError):
        evaluate_expression("x +* 1", {"x": np.array([1.0])})


def test_keyword_arguments_rejected():
    with pytest.raises(ExpressionError):
        evaluate_expression("min(1, 2, key=abs)", {})
