import math
import pytest
from littlegrad.engine import Value


# ---------- add ----------

def test_add_backward():
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    assert c.data == 5.0
    c.backward()
    assert a.grad == 1.0
    assert b.grad == 1.0


def test_add_with_number():
    a = Value(2.0)
    c = a + 3
    assert c.data == 5.0
    c.backward()
    assert a.grad == 1.0


def test_radd_number_on_left():
    a = Value(2.0)
    c = 1 + a
    assert c.data == 3.0
    c.backward()
    assert a.grad == 1.0


def test_sum_of_values():
    a, b, c = Value(1.0), Value(2.0), Value(3.0)
    d = sum([a, b, c])
    d.backward()
    assert d.data == 6.0
    assert a.grad == b.grad == c.grad == 1.0


# ---------- mul ----------

def test_mul_backward():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    assert c.data == 6.0
    c.backward()
    assert a.grad == 3.0   
    assert b.grad == 2.0


def test_square_accumulates():
    a = Value(2.0)
    c = a * a
    c.backward()
    assert a.grad == 4.0


def test_mul_with_number():
    a = Value(2.0)
    c = a * 5
    assert c.data == 10.0
    c.backward()
    assert a.grad == 5.0


def test_rmul_number_on_left():
    a = Value(2.0)
    c = 5 * a
    assert c.data == 10.0
    c.backward()
    assert a.grad == 5.0


# ---------- sub neg ----------

def test_neg_backward():
    a = Value(2.0)
    c = -a
    assert c.data == -2.0
    c.backward()
    assert a.grad == -1.0


def test_sub_backward():
    a = Value(5.0)
    b = Value(3.0)
    c = a - b
    assert c.data == 2.0
    c.backward()
    assert a.grad == 1.0
    assert b.grad == -1.0


def test_sub_with_number():
    a = Value(5.0)
    c = a - 1
    assert c.data == 4.0
    c.backward()
    assert a.grad == 1.0


def test_rsub_number_on_left():
    # 10 - a 内部是 (-a) + 10，梯度要经过中间的取负节点
    a = Value(3.0)
    c = 10 - a
    c.backward()
    assert c.data == 7.0
    assert a.grad == -1.0


# ---------- pow ----------

def test_pow_backward():
    a = Value(3.0)
    c = a ** 2
    assert c.data == 9.0
    c.backward()
    assert a.grad == 6.0   


def test_pow_negative_base():
    a = Value(-2.0)
    c = a ** -1
    assert c.data == -0.5
    c.backward()
    assert a.grad == pytest.approx(-0.25)  


def test_pow_base_one():
    a = Value(1.0)
    c = a ** 3
    assert c.data == 1.0
    c.backward()
    assert a.grad == 3.0


def test_pow_fractional_exponent():
    a = Value(4.0)
    c = a ** 0.5
    assert c.data == 2.0
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
    c.backward()
    assert a.grad == pytest.approx(1 / 3)     # 1/b
    assert b.grad == pytest.approx(-6 / 9)    # -a/b²


def test_truediv_with_number():
    a = Value(6.0)
    c = a / 2
    assert c.data == 3.0
    c.backward()
    assert a.grad == pytest.approx(0.5)


def test_rtruediv_number_on_left():
    # 2 / a 内部是 (a ** -1) * 2
    a = Value(4.0)
    c = 2 / a
    c.backward()
    assert c.data == pytest.approx(0.5)
    assert a.grad == pytest.approx(-2 / 16)   # -2/a²


# ---------- tanh ----------

def test_tanh_backward():
    a = Value(0.5)
    c = a.tanh()
    c.backward()
    assert c.data == pytest.approx(0.4621171573)
    assert a.grad == pytest.approx(0.7864477330)    # 1 - tanh(0.5)**2


def test_tanh_zero():
    a = Value(0.0)
    c = a.tanh()
    c.backward()
    assert c.data == 0.0
    assert a.grad == 1.0     # 原点处斜率为 1


@pytest.mark.parametrize("x", [-2.0, -0.5, 0.0, 0.5, 2.0])
def test_tanh_matches_math(x):
    a = Value(x)
    c = a.tanh()
    c.backward()
    assert c.data == pytest.approx(math.tanh(x))
    assert a.grad == pytest.approx(1 - math.tanh(x) ** 2)


# ---------- exp ----------

def test_exp_backward():
    a = Value(1.0)
    c = a.exp()
    c.backward()
    assert c.data == pytest.approx(math.e)
    assert a.grad == pytest.approx(math.e)    # exp 的导数是它自己


def test_exp_zero():
    a = Value(0.0)
    c = a.exp()
    c.backward()
    assert c.data == 1.0
    assert a.grad == 1.0


# ---------- log ----------

def test_log_backward():
    a = Value(2.0)
    c = a.log()
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
    c.backward()
    assert c.data == 3.0
    assert a.grad == 1.0        # 斜率是 1，不是 x 本身


def test_relu_negative():
    a = Value(-2.0)
    c = a.relu()
    c.backward()
    assert c.data == 0.0
    assert a.grad == 0.0


def test_relu_zero():
    a = Value(0.0)
    c = a.relu()
    c.backward()
    assert c.data == 0.0
    assert a.grad == 0.0        # 约定：x=0 处梯度当作 0


def test_relu_scales_upstream_gradient():
    # 上游梯度不是 1 的时候，relu 要原样传递（x > 0）或截断（x < 0）
    a = Value(3.0)
    b = Value(-3.0)
    (a.relu() * 5).backward()
    (b.relu() * 5).backward()
    assert a.grad == 5.0
    assert b.grad == 0.0


# ---------- expressions（一次 backward 自动传完整张图） ----------

def test_chain_mul_then_add():
    # d = a*b + a，a=2, b=3
    # 手算：dd/da = b + 1 = 4，dd/db = a = 2
    a = Value(2.0)
    b = Value(3.0)
    d = a * b + a
    d.backward()
    assert d.data == 8.0
    assert a.grad == 4.0
    assert b.grad == 2.0


def test_squared_error_loss():
    # loss = (pred - target)**2，pred=5, target=3
    # 手算：dloss/dpred = 2*(pred-target) = 4，dloss/dtarget = -4
    pred = Value(5.0)
    target = Value(3.0)
    loss = (pred - target) ** 2
    loss.backward()
    assert loss.data == 4.0
    assert pred.grad == 4.0
    assert target.grad == -4.0


def test_value_used_in_multiple_places():
    # y = a*a + a*3，a=2  →  dy/da = 2a + 3 = 7
    a = Value(2.0)
    y = a * a + a * 3
    y.backward()
    assert y.data == 10.0
    assert a.grad == 7.0


def test_node_used_at_different_depths():
    # a 既被 c 直接使用，又被 b 使用；拓扑顺序错了会得到 2.0
    # c = a + 3a = 4a = 8x，所以 dc/dx = 8
    x = Value(1.0)
    a = x * 2
    b = a * 3
    c = a + b
    c.backward()
    assert c.data == 8.0
    assert x.grad == 8.0


def test_exp_then_log_cancels():
    # log(exp(x)) = x，所以导数恒为 1
    a = Value(2.0)
    l = a.exp().log()
    l.backward()
    assert l.data == pytest.approx(2.0)
    assert a.grad == pytest.approx(1.0)


def test_tanh_composed_matches_direct():
    # 用 exp、减、加、除拼出 tanh，和直接写的 tanh 对比值和梯度
    x1 = Value(0.5)
    direct = x1.tanh()
    direct.backward()

    x2 = Value(0.5)
    e = (2 * x2).exp()
    composed = (e - 1) / (e + 1)
    composed.backward()

    assert composed.data == pytest.approx(direct.data)
    assert x2.grad == pytest.approx(x1.grad)