import math
import random
import pytest
from littlegrad.engine import Value
from littlegrad.nn import Neuron, Layer, MLP


def set_params(module, values):
    params = module.parameters()
    assert len(params) == len(values)
    for p, v in zip(params, values):
        p.data = v


def zero_grad(module):
    for p in module.parameters():
        p.grad = 0.0


# ---------- Neuron ----------

def test_neuron_parameter_count():
    n = Neuron(3)
    assert len(n.parameters()) == 4         
    assert all(isinstance(p, Value) for p in n.parameters())


def test_neuron_parameters_are_the_actual_weights():
    # parameters() 返回的必须是同一批对象，更新它们才会影响神经元
    n = Neuron(3)
    params = n.parameters()
    assert all(a is b for a, b in zip(params, n.w + [n.b]))


def test_neuron_init_range():
    random.seed(0)
    n = Neuron(50)
    assert all(-1 <= p.data <= 1 for p in n.parameters())


def test_neuron_forward_known_values():
    n = Neuron(2)
    n.w[0].data = 0.5
    n.w[1].data = -1.0
    n.b.data = 0.25
    out = n([2.0, 3.0])
    # 0.5*2 + (-1)*3 + 0.25 = -1.75
    assert out.data == pytest.approx(math.tanh(-1.75))


def test_neuron_zero_input_returns_tanh_of_bias():
    n = Neuron(3)
    n.b.data = 0.3
    out = n([0.0, 0.0, 0.0])
    assert out.data == pytest.approx(math.tanh(0.3))


def test_neuron_output_in_tanh_range():
    random.seed(1)
    n = Neuron(5)
    out = n([10.0, -7.0, 3.0, 0.5, 2.0])
    assert -1.0 <= out.data <= 1.0


def test_neuron_backward_known_values():
    n = Neuron(2)
    n.w[0].data = 0.5
    n.w[1].data = -1.0
    n.b.data = 0.25
    out = n([2.0, 3.0])
    out.backward()
    d = 1 - math.tanh(-1.75) ** 2           
    assert n.w[0].grad == pytest.approx(d * 2.0)
    assert n.w[1].grad == pytest.approx(d * 3.0)
    assert n.b.grad == pytest.approx(d)


def test_neuron_rejects_wrong_input_size():
    n = Neuron(3)
    with pytest.raises((AssertionError, ValueError)):
        n([1.0, 2.0])
    with pytest.raises((AssertionError, ValueError)):
        n([1.0, 2.0, 3.0, 4.0])


# ---------- Layer ----------

def test_layer_parameter_count():
    layer = Layer(3, 4)
    assert len(layer.parameters()) == 4 * (3 + 1)


def test_layer_output_size():
    layer = Layer(3, 4)
    out = layer([1.0, 2.0, 3.0])
    assert len(out) == 4


def test_layer_neurons_have_independent_parameters():
    layer = Layer(3, 4)
    params = layer.parameters()
    assert len({id(p) for p in params}) == len(params)


def test_layer_forward_known_values():
    # 两个神经元收到同一份输入，各自用自己的权重
    layer = Layer(2, 2)
    set_params(layer, [0.5, -1.0, 0.25,      # 神经元 1：w0, w1, b
                       1.0, 1.0, 0.0])       # 神经元 2：w0, w1, b
    out = layer([2.0, 3.0])
    assert out[0].data == pytest.approx(math.tanh(0.5 * 2 - 1.0 * 3 + 0.25))
    assert out[1].data == pytest.approx(math.tanh(1.0 * 2 + 1.0 * 3 + 0.0))


def test_layer_rejects_wrong_input_size():
    layer = Layer(3, 2)
    with pytest.raises((AssertionError, ValueError)):
        layer([1.0, 2.0])


# ---------- MLP ----------

def test_mlp_parameter_count():
    mlp = MLP(3, [4, 4, 1])
    assert len(mlp.parameters()) == 41        


def test_mlp_layer_sizes():
    mlp = MLP(3, [4, 4, 1])
    assert len(mlp.layers) == 3
    assert [len(layer.parameters()) for layer in mlp.layers] == [16, 20, 5]


@pytest.mark.parametrize("nouts", [[1], [4, 1], [4, 4, 3]])
def test_mlp_output_size_matches_last_layer(nouts):
    mlp = MLP(3, nouts)
    out = mlp([1.0, 2.0, 3.0])
    assert len(out) == nouts[-1]


def test_mlp_forward_known_values():
    # 所有权重和偏置都设成 0.5，x=[1, 2]，MLP(2, [2, 1])
    # 第一层每个神经元：0.5*1 + 0.5*2 + 0.5 = 2.0 → t = tanh(2)
    # 第二层：0.5*t + 0.5*t + 0.5 = t + 0.5 → tanh(t + 0.5)
    mlp = MLP(2, [2, 1])
    set_params(mlp, [0.5] * len(mlp.parameters()))
    out = mlp([1.0, 2.0])
    t = math.tanh(2.0)
    assert out[0].data == pytest.approx(math.tanh(t + 0.5))


def test_mlp_rejects_wrong_input_size():
    mlp = MLP(3, [4, 1])
    with pytest.raises((AssertionError, ValueError)):
        mlp([1.0, 2.0])


def test_mlp_backward_reaches_every_parameter():
    mlp = MLP(2, [2, 1])
    set_params(mlp, [0.5] * len(mlp.parameters()))
    out = mlp([1.0, 2.0])[0]
    out.backward()
    assert all(p.grad != 0.0 for p in mlp.parameters())


def test_mlp_gradients_match_numerical_gradients():

    random.seed(0)
    mlp = MLP(3, [4, 4, 1])
    x = [2.0, 3.0, -1.0]

    out = mlp(x)[0]
    out.backward()
    analytic = [p.grad for p in mlp.parameters()]

    h = 1e-6
    for p, g in zip(mlp.parameters(), analytic):
        original = p.data
        p.data = original + h
        f_plus = mlp(x)[0].data
        p.data = original - h
        f_minus = mlp(x)[0].data
        p.data = original
        numeric = (f_plus - f_minus) / (2 * h)
        assert g == pytest.approx(numeric, rel=1e-4, abs=1e-8)


# ---------- 训练相关 ----------

def test_grads_accumulate_across_fresh_forward_passes():
    # 每一步重新前向会建一张新图，但参数是同一批对象，
    # 不清零的话参数的 grad 会把两步的梯度加在一起
    random.seed(0)
    mlp = MLP(2, [3, 1])
    x1, x2 = [1.0, 2.0], [-1.0, 0.5]

    def grads_for(x):
        zero_grad(mlp)
        mlp(x)[0].backward()
        return [p.grad for p in mlp.parameters()]

    g1 = grads_for(x1)
    g2 = grads_for(x2)

    zero_grad(mlp)
    mlp(x1)[0].backward()
    mlp(x2)[0].backward()                    
    summed = [p.grad for p in mlp.parameters()]

    for s, a, b in zip(summed, g1, g2):
        assert s == pytest.approx(a + b)


def test_one_gradient_step_reduces_loss():
    random.seed(0)
    mlp = MLP(3, [4, 4, 1])
    x, target = [2.0, 3.0, -1.0], 1.0

    loss0 = (mlp(x)[0] - target) ** 2
    zero_grad(mlp)
    loss0.backward()
    for p in mlp.parameters():
        p.data += -0.01 * p.grad

    loss1 = (mlp(x)[0] - target) ** 2
    assert loss1.data < loss0.data


def test_mlp_learns_tiny_dataset():
    random.seed(0)
    xs = [[2.0, 3.0, -1.0],
          [3.0, -1.0, 0.5],
          [0.5, 1.0, 1.0],
          [1.0, 1.0, -1.0]]
    ys = [1.0, -1.0, -1.0, 1.0]
    mlp = MLP(3, [4, 4, 1])

    for step in range(200):
        preds = [mlp(x)[0] for x in xs]
        loss = sum((p - y) ** 2 for p, y in zip(preds, ys))
        zero_grad(mlp)
        loss.backward()
        for p in mlp.parameters():
            p.data += -0.05 * p.grad

    preds = [mlp(x)[0].data for x in xs]
    assert loss.data < 0.05
    assert all(p * y > 0 for p, y in zip(preds, ys))  