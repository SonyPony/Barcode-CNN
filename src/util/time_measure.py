# coding=utf-8
from time import time


class measure_time():
    def __init__(self, print_format=None):
        self._print_format = print_format

    def __enter__(self):
        self._start = time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._t = time() - self._start
        if self._print_format:
            print(self._print_format.format(self.time))

    @property
    def time(self) -> float:
        return self._t