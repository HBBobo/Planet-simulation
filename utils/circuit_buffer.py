import numpy as np
from numpy.typing import NDArray
from typing import Sequence

class CircularBuffer:

    size: int
    dimension: int
    data: NDArray
    index: int
    full: bool

    def __init__(self, size: int, dimension: int):
        """
        Initializes the CircularBuffer with a given size.

        Args:
            size (int): The size of the buffer.
            dimension (int): The number of dimensions of the data.
        """

        self.size = size
        self.dimension = dimension
        self.data = np.zeros((size, self.dimension))
        self.index = 0
        self.full = False


    def append(self, element: Sequence[float] | float):
        """
        Adds a new element to the buffer.
        Accepts scalar for 1D, or list/array for nD.

        Args:
            element (Sequence[float] | float): The element to add to the buffer.
        """

        if self.dimension == 1 and not isinstance(element, (list, tuple, np.ndarray)):
            element = [element]

        element = np.asarray(element)
        if element.shape != (self.dimension,):
            raise ValueError(f"Element must have shape ({self.dimension},), got {element.shape}")

        self.data[self.index] = element
        self.index = (self.index + 1) % self.size
        if self.index == 0:
            self.full = True


    def get(self) -> NDArray:
        """
        Returns the contents of the buffer in order.
        """
        if self.full:
            return np.concatenate((self.data[self.index:], self.data[:self.index]))
        else:
            return self.data[:self.index]


    def __len__(self):
        return self.size if self.full else self.index