from .Device import Device

__all__ = ['mps']


class mps(Device):
    def __init__(self):
        super().__init__(
            name="mps",
        )