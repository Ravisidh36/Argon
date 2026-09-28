import ast
import operator
import math

# Deliberately NOT using eval() here — eval() would let an LLM-controlled
# string run arbitrary Python, which is a real code-execution risk in any
# agent. Instead we parse the expression into a syntax tree and only allow
# this fixed whitelist of operators/functions to actually execute.

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_ALLOWED_FUNCTIONS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
}

_ALLOWED_NAMES = {
    "pi": math.pi,
    "e": math.e,
}


def _eval_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError(f"Operator {op_type.__name__} is not allowed")
        return _ALLOWED_OPERATORS[op_type](
            _eval_node(node.left), _eval_node(node.right)
        )

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError(f"Operator {op_type.__name__} is not allowed")
        return _ALLOWED_OPERATORS[op_type](_eval_node(node.operand))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCTIONS:
            raise ValueError("Only whitelisted math functions are allowed")
        args = [_eval_node(arg) for arg in node.args]
        return _ALLOWED_FUNCTIONS[node.func.id](*args)

    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_NAMES:
            return _ALLOWED_NAMES[node.id]
        raise ValueError(f"Unknown name: {node.id}")

    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def calculate(expression):
    """Safely evaluates a math expression string, e.g. '2 * (3 + 4)'."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


calculate_tool = {
    "name": "calculate",
    "description": (
        "Evaluates a mathematical expression and returns the result. "
        "Supports +, -, *, /, %, ** (power), and functions sqrt, sin, cos, "
        "tan, log, log10, abs, round, floor, ceil, plus constants pi and e."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The math expression to evaluate, e.g. '2 * (3 + 4)' or 'sqrt(16) + pi'",
            }
        },
        "required": ["expression"],
    },
}
