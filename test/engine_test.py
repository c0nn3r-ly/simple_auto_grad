import pytest
from core.engine import Value


# ---------- 加法 ----------

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
    # sum() 从数字 0 开始，实际会先执行 0 + a，依赖 __radd__
    a, b, c = Value(1.0), Value(2.0), Value(3.0)
    d = sum([a, b, c])
    assert d.data == 6.0


# ---------- 乘法 ----------

def test_mul_backward():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    assert c.data == 6.0
    c.grad = 1.0
    c.backward()
    assert a.grad == 3.0   # 局部导数是对方的 data
    assert b.grad == 2.0


def test_square_accumulates():
    # self 和 other 是同一个对象，梯度必须累加：d(a*a)/da = 2a = 4
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


# ---------- 取负、减法 ----------

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
    # 减法由 a + (b * -1) 拼成，b 的梯度要经过中间节点，
    # 完整检查见下面的 test_squared_error_loss
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


# ---------- 乘方 ----------

def test_pow_backward():
    a = Value(3.0)
    c = a ** 2
    assert c.data == 9.0
    c.grad = 1.0
    c.backward()
    assert a.grad == 6.0   # 2 * 3


def test_pow_negative_base():
    a = Value(-2.0)
    c = a ** -1
    assert c.data == -0.5
    c.grad = 1.0
    c.backward()
    assert a.grad == pytest.approx(-0.25)   # -1 * (-2)^-2


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
    assert a.grad == pytest.approx(0.25)   # 0.5 * 4^(-0.5)


def test_pow_rejects_value_exponent():
    a = Value(2.0)
    with pytest.raises(AssertionError):
        a ** Value(2.0)


# ---------- 多步表达式（手动按逆序调用 backward） ----------

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
    # loss = (pred - target)**2，pred=5, target=3
    # 手算：dloss/dpred = 2*(pred-target) = 4，dloss/dtarget = -4
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
    # y = a*a + a*3，a=2  →  dy/da = 2a + 3 = 7
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
