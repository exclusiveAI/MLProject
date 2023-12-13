__all__ = ["Metric"]

class Metric:
    """
    A class that represents a metric
    Args:
        name (str): The name of the metric
        f (function): The function that calculates the metric
    Attributes:
        name (str): The name of the metric
        f (function): The function that calculates the metric
    """
    def __init__(self, name, f):
        self.name = name
        self.f = f

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)


