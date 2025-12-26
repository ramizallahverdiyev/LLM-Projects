# tests/test_calculator.py

import pytest
from src.tools.calculator import evaluate_expression
import math

def test_add():
    res = evaluate_expression("2 + 3")
    assert res.get("result") == 5

def test_subtract():
    res = evaluate_expression("10 - 4")
    assert res.get("result") == 6

def test_multiply():
    res = evaluate_expression("6 * 7")
    assert res.get("result") == 42

def test_divide():
    res = evaluate_expression("20 / 4")
    assert res.get("result") == 5

def test_power():
    res = evaluate_expression("2 ** 3")
    assert res.get("result") == 8

def test_sqrt():
    res = evaluate_expression("sqrt(16)")
    assert res.get("result") == 4

def test_sin():
    # sin(pi/2) ≈ 1
    res = evaluate_expression("sin(3.14159265 / 2)")
    assert abs(res.get("result") - 1) < 1e-6

def test_cos():
    res = evaluate_expression("cos(0)")
    assert abs(res.get("result") - 1) < 1e-6

def test_error_handling():
    res = evaluate_expression("unknown_func(5)")
    assert "error" in res
