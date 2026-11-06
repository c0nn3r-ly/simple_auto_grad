import math
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


def test_rsub_number_on_left():
    a = Value(3.0)
    c = 10 - a
    assert c.data == 7.0


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


# ---------- truediv ----------

def test_truediv_value():
    a = Value(6.0)
    b = Value(3.0)
    c = a / b
    assert c.data == pytest.approx(2.0)
    c.grad = 1.0
    c.backward()
    assert a.grad == pytest.approx(1 / 3)     


def test_truediv_with_number():
    a = Value(6.0)
    c = a / 2
    assert c.data == 3.0
    c.grad = 1.0
    c.backward()
    assert a.grad == pytest.approx(0.5)


def test_rtruediv_number_on_left():
    a = Value(4.0)
    c = 2 / a
    assert c.data == pytest.approx(0.5)


# ---------- tanh ----------

def test_tanh_backward():
    a = Value(0.5)
    c = a.tanh()
    c.grad = 1.0
    c.backward()
    assert c.data == pytest.approx(0.4621171573)
    assert a.grad == pytest.approx(0.7864477330)    # 1 - tanh(0.5)**2


def test_tanh_zero():
    a = Value(0.0)
    c = a.tanh()
    c.grad = 1.0
    c.backward()
    assert c.data == 0.0
    assert a.grad == 1.0     


@pytest.mark.parametrize("x", [-2.0, -0.5, 0.0, 0.5, 2.0])
def test_tanh_matches_math(x):
    a = Value(x)
    c = a.tanh()
    c.grad = 1.0
    c.backward()
    assert c.data == pytest.approx(math.tanh(x))
    assert a.grad == pytest.approx(1 - math.tanh(x) ** 2)


# ---------- exp ----------

def test_exp_backward():
    a = Value(1.0)
    c = a.exp()
    c.grad = 1.0
    c.backward()
    assert c.data == pytest.approx(math.e)
    assert a.grad == pytest.approx(math.e)    


def test_exp_zero():
    a = Value(0.0)
    c = a.exp()
    c.grad = 1.0
    c.backward()
    assert c.data == 1.0
    assert a.grad == 1.0


# ---------- log ----------

def test_log_backward():
    a = Value(2.0)
    c = a.log()
    c.grad = 1.0
    c.backward()
    assert c.data == pytest.approx(math.log(2))
    assert a.grad == pytest.approx(0.5)     # 1/x


@pytest.mark.parametrize("x", [0.0, -1.0])
def test_log_rejects_nonpositive(x):
    with pytest.raises(ValueError):
        Value(x).log()


# ---------- relu ----------

def test_relu_positive():
    a = Value(3.0)
    c = a.relu()
    c.grad = 1.0
    c.backward()
    assert c.data == 3.0
    assert a.grad == 1.0        


def test_relu_negative():
    a = Value(-2.0)
    c = a.relu()
    c.grad = 1.0
    c.backward()
    assert c.data == 0.0
    assert a.grad == 0.0


def test_relu_zero():
    a = Value(0.0)
    c = a.relu()
    c.grad = 1.0
    c.backward()
    assert c.data == 0.0
    assert a.grad == 0.0        


def test_relu_scales_upstream_gradient():
    a = Value(3.0)
    b = Value(-3.0)
    ca = a.relu()
    cb = b.relu()
    ca.grad = 5.0
    cb.grad = 5.0
    ca.backward()
    cb.backward()
    assert a.grad == 5.0
    assert b.grad == 0.0


# ---------- expressions test manually ----------

def test_chain_mul_then_add():
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


def test_exp_then_log_cancels():
    a = Value(2.0)
    e = a.exp()
    l = e.log()
    l.grad = 1.0
    l.backward()
    e.backward()
    assert l.data == pytest.approx(2.0)
    assert a.grad == pytest.approx(1.0)


needs_topo = pytest.mark.skip(reason="")


@needs_topo
def test_chain_auto():
    a = Value(2.0)
    b = Value(3.0)
    d = a * b + a
    d.backward()
    assert d.data == 8.0
    assert a.grad == 4.0
    assert b.grad == 2.0


@needs_topo
def test_value_used_in_multiple_places_auto():
    a = Value(2.0)
    y = a * a + a * 3
    y.backward()
    assert y.data == 10.0
    assert a.grad == 7.0


@needs_topo
def test_squared_error_loss_auto():
    pred = Value(5.0)
    target = Value(3.0)
    loss = (pred - target) ** 2
    loss.backward()
    assert loss.data == 4.0
    assert pred.grad == 4.0
    assert target.grad == -4.0


@needs_topo
def test_rsub_backward_auto():
    a = Value(3.0)
    c = 10 - a
    c.backward()
    assert c.data == 7.0
    assert a.grad == -1.0


@needs_topo
def test_truediv_backward_auto():
    a = Value(6.0)
    b = Value(3.0)
    c = a / b
    c.backward()
    assert a.grad == pytest.approx(1 / 3)       
    assert b.grad == pytest.approx(-6 / 9)      


@needs_topo
def test_rtruediv_backward_auto():
    a = Value(4.0)
    c = 2 / a
    c.backward()
    assert c.data == pytest.approx(0.5)
    assert a.grad == pytest.approx(-2 / 16)     


@needs_topo
def test_tanh_composed_matches_direct():
    x1 = Value(0.5)
    direct = x1.tanh()
    direct.backward()

    x2 = Value(0.5)
    e = (2 * x2).exp()
    composed = (e - 1) / (e + 1)
    composed.backward()

    assert composed.data == pytest.approx(direct.data)
    assert x2.grad == pytest.approx(x1.grad)