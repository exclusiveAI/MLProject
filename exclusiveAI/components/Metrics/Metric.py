__all__ = ["Metric"]

class Metric:
    def __init__(self, name, f):
        self.name = name
        self.f = f
        