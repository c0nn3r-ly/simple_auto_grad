import pytest
from littlegrad.engine import Value


# ---------- add ----------

def test_add_backward():
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    assert c.data == 5.0
    c.grad = 1.0
    c.backward()
    assert a.grad == 1.0
    assert b.grad == 1.0


def test_add_with_number():
    a = Value(2.0)
    c = a + 3
    assert c.data == 5.0
    c.grad = 1.0
    c.backward()
    assert a.grad == 1.0


def test_radd_number_on_left():
    a = Value(2.0)
    c = 1 + a
    assert c.data == 3.0
    c.grad = 1.0
    c.backward()
    assert a.grad == 1.0


def test_sum_of_values():
    a, b, c = Value(1.0), Value(2.0), Value(3.0)
    d = sum([a, b, c])
    assert d.data == 6.0


# ---------- mul ----------

def test_mul_backward():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    assert c.data == 6.0
    c.grad = 1.0
    c.backward()
    assert a.grad == 3.0   
    assert b.grad == 2.0


def test_square_accumulates():
    a = Value(2.0)
    c = a * a
    c.grad = 1.0
    c.backward()
    assert a.grad == 4.0


def test_mul_with_number():
    a = Value(2.0)
    c = a * 5
    assert c.data == 10.0
    c.grad = 1.0
    c.backward()
    assert a.grad == 5.0


def test_rmul_number_on_left():
    a = Value(2.0)
    c = 5 * a
    assert c.data == 10.0
    c.grad = 1.0
    c.backward()
    assert a.grad == 5.0


# ---------- sub neg ----------

def test_neg_backward():
    a = Value(2.0)
    c = -a
    assert c.data == -2.0
    c.grad = 1.0
    c.backward()
    assert a.grad == -1.0


def test_sub_backward():
    a = Value(5.0)
    b = Value(3.0)
    c = a - b
    assert c.data == 2.0
    c.grad = 1.0
    c.backward()
    assert a.grad == 1.0


def test_sub_with_number():
    a = Value(5.0)
    c = a - 1
    assert c.data == 4.0
    c.grad = 1.0
    c.backward()
    assert a.grad == 1.0


# ---------- pow ----------

def test_pow_backward():
    a = Value(3.0)
    c = a ** 2
    assert c.data == 9.0
    c.grad = 1.0
    c.backward()
    assert a.grad == 6.0   


def test_pow_negative_base():
    a = Value(-2.0)
    c = a ** -1
    assert c.data == -0.5
    c.grad = 1.0
    c.backward()
    assert a.grad == pytest.approx(-0.25)  


def test_pow_base_one():
    a = Value(1.0)
    c = a ** 3
    assert c.data == 1.0
    c.grad = 1.0
    c.backward()
    assert a.grad == 3.0


def test_pow_fractional_exponent():
    a = Value(4.0)
    c = a ** 0.5
    assert c.data == 2.0
    c.grad = 1.0
    c.backward()
    assert a.grad == pytest.approx(0.25)   


def test_pow_rejects_value_exponent():
    a = Value(2.0)
    with pytest.raises(AssertionError):
        a ** Value(2.0)


# ---------- expressions test manually ----------

def test_chain_mul_then_add():
    # d = a*b + a，a=2, b=3
    # 手算：dd/da = b + 1 = 4，dd/db = a = 2
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    d = c + a
    d.grad = 1.0
    d.backward()
    c.backward()
    assert d.data == 8.0
    assert a.grad == 4.0
    assert b.grad == 2.0


def test_squared_error_loss():
    pred = Value(5.0)
    target = Value(3.0)
    neg_target = -target
    diff = pred + neg_target
    loss = diff ** 2
    loss.grad = 1.0
    loss.backward()
    diff.backward()
    neg_target.backward()
    assert loss.data == 4.0
    assert pred.grad == 4.0
    assert target.grad == -4.0


def test_value_used_in_multiple_places():
    a = Value(2.0)
    p = a * a
    q = a * 3
    y = p + q
    y.grad = 1.0
    y.backward()
    q.backward()
    p.backward()
    assert y.data == 10.0
    assert a.grad == 7.0
