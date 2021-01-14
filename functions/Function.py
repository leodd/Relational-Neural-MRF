from abc import ABC, abstractmethod


class Function(ABC):
    @abstractmethod
    def __call__(self, *parameters):
        """A method return a function value.

        Given an assignment to the parameters of the function, return a value of the function.

        Args:
            *parameters: A list of assignment, e.g. [1, 2, 3].

        Returns: A number value.

        """
        pass

    def slice(self, *parameters):
        """A method that creates a slice function.

        Given an assignment to the subset of the parameters, return a callable Function class instance,
        where the un-assigned variable would be the new parameters.

        Args:
            *parameters: A list of assignment, for un-assigned variables, set it to None, e.g. [None, 1, 2, 3].

        Returns: A Function class instance.

        """
        pass

    def __add__(self, other):
        pass

    def __sub__(self, other):
        pass

    def __neg__(self):
        pass

    def __mul__(self, other):
        pass

    def __truediv__(self, other):
        pass
