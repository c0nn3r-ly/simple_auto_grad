import math

class Value:
   
    def __init__(self, data, _children=(), _op='', label='', grad=0):
        self.data = data 
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = grad
        self.backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        def backward():
            other.grad += out.grad
            self.grad += out.grad
        out.backward = backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        def backward():
            other.grad += out.grad * self.data
            self.grad += out.grad * other.data
        out.backward = backward
        return out
