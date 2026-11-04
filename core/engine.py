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
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def backward():
            other.grad += out.grad
            self.grad += out.grad
        out.backward = backward
        return out

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other
    
    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def backward():
            other.grad += out.grad * self.data
            self.grad += out.grad * other.data
        out.backward = backward
        return out
    
    def __neg__(self):
        return self * -1
    
    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Value(self.data ** other, (self,), '**')
        def backward():
            self.grad += out.grad * other * (self.data ** (other - 1))
        out.backward = backward
        return out

    def __truediv__(self, other):
        return self * (other ** -1)
    
    def __rtruediv__(self, other):
        return (self ** -1) * other