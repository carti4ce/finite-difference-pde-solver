"""Safe evaluation of user-supplied math expressions (source terms, nonlinear
reaction terms, initial conditions).

User expressions come from an HTTP request body, so this deliberately does
**not** use Python's ``eval``/``exec`` on the input. Instead the expression is
parsed to an AST and walked by hand, allowing only a small whitelist of node
types, names, and functions; anything else (attribute access, subscripting,
comprehensions, lambdas, walrus, calls to unknown names, ...) is rejected
before any computation happens. This closes off ``__import__``, dunder
attribute lookups, and similar sandbox-escape tricks that plague naive
``eval(expr, {"__builtins__": {}})`` approaches.

Expressions are evaluated with NumPy so they vectorize over grid arrays.
"""
from __future__ import annotations

import ast
import math
import operator

import numpy as np

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}

_UNARYOPS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}

_FUNCTIONS = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "atan2": np.arctan2,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "sign": np.sign,
    "floor": np.floor,
    "ceil": np.ceil,
    "min": np.minimum,
    "max": np.maximum,
    "where": np.where,
    "clip": np.clip,
}


class ExpressionError(ValueError):
    """An expression failed to parse or referenced something not allowed."""


def _describe(node: ast.AST) -> str:
    return type(node).__name__


def _eval(node: ast.AST, variables: dict):
    if isinstance(node, ast.Expression):
        return _eval(node.body, variables)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ExpressionError(f"unsupported literal: {node.value!r}")

    if isinstance(node, ast.Name):
        if node.id in variables:
            return variables[node.id]
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ExpressionError(
            f"unknown name {node.id!r} (available: "
            f"{', '.join(sorted(set(variables) | set(_CONSTANTS)))})"
        )

    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise ExpressionError(f"unsupported operator: {_describe(node.op)}")
        return op(_eval(node.left, variables), _eval(node.right, variables))

    if isinstance(node, ast.UnaryOp):
        op = _UNARYOPS.get(type(node.op))
        if op is None:
            raise ExpressionError(f"unsupported unary operator: {_describe(node.op)}")
        return op(_eval(node.operand, variables))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ExpressionError("only calls to a whitelisted function name are allowed")
        if node.keywords:
            raise ExpressionError("keyword arguments are not supported")
        fn = _FUNCTIONS.get(node.func.id)
        if fn is None:
            raise ExpressionError(
                f"unknown function {node.func.id!r} (available: "
                f"{', '.join(sorted(_FUNCTIONS))})"
            )
        args = [_eval(a, variables) for a in node.args]
        return fn(*args)

    if isinstance(node, ast.IfExp):
        # ternary: <body> if <test> else <orelse> — vectorized via np.where
        return np.where(_eval(node.test, variables),
                        _eval(node.body, variables),
                        _eval(node.orelse, variables))

    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise ExpressionError("chained comparisons are not supported")
        left = _eval(node.left, variables)
        right = _eval(node.comparators[0], variables)
        cmp = {
            ast.Lt: operator.lt, ast.LtE: operator.le,
            ast.Gt: operator.gt, ast.GtE: operator.ge,
            ast.Eq: operator.eq, ast.NotEq: operator.ne,
        }.get(type(node.ops[0]))
        if cmp is None:
            raise ExpressionError(f"unsupported comparison: {_describe(node.ops[0])}")
        return cmp(left, right)

    raise ExpressionError(f"unsupported syntax: {_describe(node)}")


def compile_expression(expression: str, allowed_variables: set) -> ast.Expression:
    """Parse ``expression`` and verify it only touches whitelisted names.

    Raises `ExpressionError` for anything outside the numeric-expression
    subset (imports, attribute/subscript access, comprehensions, lambdas,
    assignments, unknown names/functions, ...).
    """
    if not expression or not expression.strip():
        raise ExpressionError("expression must not be empty")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"invalid syntax: {exc.msg}") from exc

    allowed_names = set(allowed_variables) | set(_CONSTANTS)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            if node.id in _FUNCTIONS:
                continue  # validated at call sites below
            raise ExpressionError(f"unknown name {node.id!r}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ExpressionError("only calls to a whitelisted function name are allowed")
            if node.func.id not in _FUNCTIONS:
                raise ExpressionError(f"unknown function {node.func.id!r}")
    return tree


def evaluate_expression(expression: str, variables: dict):
    """Compile and evaluate ``expression`` against ``variables`` (numpy-vectorized).

    Raises `ExpressionError` on anything not in the numeric-expression subset.
    """
    tree = compile_expression(expression, set(variables))
    return _eval(tree, variables)
