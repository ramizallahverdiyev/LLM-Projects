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

ALLOWED_FUNCS = {name: getattr(math, name) for name in dir(math) if not name.startswith("_")}

def evaluate_expression(expr: str) -> Dict[str, Union[float, str]]:
    try:
        node = ast.parse(expr, mode='eval').body
        result = _eval(node)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

def _eval(node) -> float:
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type}")
        return OPERATORS[op_type](_eval(node.left), _eval(node.right))
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type}")
        return OPERATORS[op_type](_eval(node.operand))
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise TypeError("Only simple function calls allowed")
        func_name = node.func.id
        if func_name not in ALLOWED_FUNCS:
            raise ValueError(f"Function {func_name} is not allowed")
        func = ALLOWED_FUNCS[func_name]
        args = [_eval(arg) for arg in node.args]
        return func(*args)
    else:
        raise TypeError(f"Unsupported expression: {node}")