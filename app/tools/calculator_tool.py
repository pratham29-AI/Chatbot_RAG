"""
Tool 2: Safe Mathematical Calculator.

Uses Python's `ast` module to parse and evaluate arithmetic expressions
without calling `eval()`.  Only whitelisted operators and math functions are
allowed, making it safe against code injection.

Supported:
  - Arithmetic:  +  -  *  /  //  %  ** (power)
  - Unary:       -x  +x
  - Functions:   sqrt, sin, cos, tan, log, log10, abs, round, ceil, floor
  - Constants:   pi, e
  - Grouping:    parentheses
"""

import ast
import math
import operator
from typing import Union

from langchain_core.tools import tool

# ── allowed operators ─────────────────────────────────────────────────────────

_BINARY_OPS: dict[type, callable] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type, callable] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_NAMES: dict[str, Union[float, callable]] = {
    # constants
    "pi": math.pi,
    "e": math.e,
    # functions
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "abs": abs,
    "round": round,
    "ceil": math.ceil,
    "floor": math.floor,
}


# ── AST evaluator ─────────────────────────────────────────────────────────────

def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported literal: {node.value!r}")

    if isinstance(node, ast.Name):
        if node.id in _NAMES:
            return _NAMES[node.id]
        raise ValueError(f"Unknown identifier: '{node.id}'")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BINARY_OPS:
            raise ValueError(f"Operator not allowed: {op_type.__name__}")
        left = _eval(node.left)
        right = _eval(node.right)
        return _BINARY_OPS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ValueError(f"Unary operator not allowed: {op_type.__name__}")
        return _UNARY_OPS[op_type](_eval(node.operand))

    if isinstance(node, ast.Call):
        func = _eval(node.func)
        if not callable(func):
            raise ValueError("Expression is not callable.")
        args = [_eval(a) for a in node.args]
        return func(*args)

    raise ValueError(f"Unsupported expression type: {type(node).__name__}")


def _safe_eval(expression: str) -> float:
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression syntax: {exc}") from exc
    return _eval(tree.body)


# ── LangChain tool ────────────────────────────────────────────────────────────

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the numeric result.

    Use this tool whenever the user asks for a calculation, even simple
    arithmetic, to guarantee accuracy.

    Supported operations: +, -, *, /, // (floor div), % (modulo), ** (power),
    and functions: sqrt, sin, cos, tan, log, log10, abs, round, ceil, floor.
    Constants: pi, e.

    Args:
        expression: A mathematical expression as a string,
                    e.g. "sqrt(144) + 2 * (7 - 3)" or "log(100) / log(10)".

    Returns:
        The computed result as a string, or an error message.
    """
    try:
        result = _safe_eval(expression)
    except (ValueError, ZeroDivisionError, OverflowError) as exc:
        return f"Calculation error: {exc}"

    # Format: strip unnecessary .0 for integers, otherwise 6 sig-figs
    if result == int(result) and abs(result) < 1e15:
        return f"{expression} = {int(result)}"
    return f"{expression} = {result:.6g}"
