import ast
import operator
import math
from typing import Union, Dict

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Allowed functions from math module
ALLOWED_FUNCS = {name: getattr(math, name) for name in dir(math) if not name.startswith("_")}

def evaluate_expression(expr: str) -> Dict[str, Union[float, str]]:
    """
    Safely evaluate a more advanced arithmetic expression using math functions.

    Parameters:
        expr (str): The arithmetic expression, e.g., "sqrt(16) + 2 ** 3"

    Returns:
        dict: {"result": value} on success, {"error": message} on failure
    """
    try:
        node = ast.parse(expr, mode='eval').body
        result = _eval(node)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

def _eval(node) -> float:
    """Recursively evaluate AST nodes with support for operators and math functions."""
    if isinstance(node, ast.Num):  # <number>
        return node.n
    elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
        op_type = type(node.op)
        if op_type not in OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type}")
        left = _eval(node.left)
        right = _eval(node.right)
        return OPERATORS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):  # - <operand> or + <operand>
        op_type = type(node.op)
        if op_type not in OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type}")
        operand = _eval(node.operand)
        return OPERATORS[op_type](operand)
    elif isinstance(node, ast.Call):  # function calls
        if not isinstance(node.func, ast.Name):
            raise TypeError("Only simple function calls allowed")
        func_name = node.func.id
        if func_name not in ALLOWED_FUNCS:
            raise ValueError(f"Function {func_name} is not allowed")
        func = ALLOWED_FUNCS[func_name]
        args = [_eval(arg) for arg in node.args]
        return func(*args)
    elif isinstance(node, ast.Expr):
        return _eval(node.value)
    else:
        raise TypeError(f"Unsupported expression: {node}")