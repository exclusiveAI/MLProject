__all__ = ['Layer']


class Layer:
    def __init__(self):
        self.next = None
        self.prev = None
        self.input = None
        self.output = None
        self.params = None
        self.grads = None
        self.name = None
        self.type = None
        self.trainable = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError
