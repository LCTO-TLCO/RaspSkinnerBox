from abc import ABCMeta, abstractmethod


class Model(metaclass=ABCMeta):
    def __init__(self):
        self.alpha = None
        self.beta = None
        self.log = None

    def read(self):
        pass

    def update(self):
        pass

    def predict(self):
        pass

    def likelihood(self):
        pass

    def likelihood_all(self):
        pass
