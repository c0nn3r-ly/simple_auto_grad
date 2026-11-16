from littlegrad.engine import Value
from littlegrad.nn import Neuron, Layer, MLP

#simple training example

n = MLP(3, [4, 4, 1])

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]

#targets
ys = [1.0, -1.0, -1.0, 1.0]

for k in range(1000):
    learning_rate = 0.1
    if(k > 900):
        learning_rate = 0.01
    #forward pass
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    
    #clear grads
    for p in n.parameters():
        p.grad = 0.0
    
    #backward pass
    loss.backward()

    #update weights
    for p in n.parameters():
        p.data += -learning_rate * p.grad

    if (k % 50 == 0):
        print(k, loss.data)

[n(x) for x in xs]
print([(i, ypred[i].data) for i, y in enumerate(ypred)])
