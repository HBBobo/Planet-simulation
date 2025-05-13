import numpy as np
from numpy.typing import NDArray


class Satellite:

    name: str
    mass: float
    position: NDArray[np.float64]
    velocity: NDArray[np.float64]
    position_history: NDArray[np.float64]
    firstIndex: int                       # First index for storing data
    lastIndex: int                        # Next free index for storing data
    maxIndex: int                         # Number of stored data points


    def __init__(self, name: str, mass: float, pos: NDArray[np.float64], vel: NDArray[np.float64], datapoints: int = 1024):
        """
        Initialization of the Satellite class.

        Args:
            mass (int): Mass of the satellite in kg.
            pos (NDArray[np.float64]): Initial position of the satellite as a 2D array.
            vel (NDArray[np.float64]): Initial velocity of the satellite as a 2D array.
            datapoints (int): Number of stored data points.

        Raises:
            ValueError: If the shape of pos or vel is not (2,).
        """

        if pos.shape != (2,):
            raise ValueError(f"Initial position must be a 2-dimensional array, but got {pos.shape}")
        if vel.shape != (2,):
            raise ValueError(f"Initial velocity must be a 2-dimensional array, but got {vel.shape}")

        self.name = name
        self.mass = mass
        self.position = pos
        self.velocity = vel

        self.position_history = np.zeros((datapoints, 2), dtype=np.float64)
        self.position_history[0] = self.position.copy()
        self.firstIndex = 0
        self.lastIndex = 1
        self.maxIndex = datapoints


    def take_input(self):
        """
        Takes user input for mass, initial position, and initial velocity of the satellite.
        Overwrites current satellite's state.
        """

        self.name = input("Enter satellite name: ")
        self.mass = float(input(f"Please give {self.name} a mass [kg]: "))

        if not isinstance(self.position, np.ndarray) or self.position.shape != (2,):
            self.position = np.zeros(2, dtype=np.float64)
        if not isinstance(self.velocity, np.ndarray) or self.velocity.shape != (2,):
            self.velocity = np.zeros(2, dtype=np.float64)

        self.position[0] = float(input(f"Please give {self.name} initial x position: "))
        self.position[1] = float(input(f"Please give {self.name} initial y position: "))
        self.velocity[0] = float(input(f"Please give {self.name} initial x velocity: "))
        self.velocity[1] = float(input(f"Please give {self.name} initial y velocity: "))
    

    def __str__(self):
        """
        Representation of the Satellite object

        Returns:
            str: A string representation of the Satellite object.
        """

        pos_str = "N/A"
        vel_str = "N/A"
        if self.position is not None and len(self.position) >= 2:
            pos_str = f"[{self.position[0]:.2f}, {self.position[1]:.2f}] m"
        if self.velocity is not None and len(self.velocity) >= 2:
            vel_str = f"[{self.velocity[0]:.2f}, {self.velocity[1]:.2f}] m/s"

        return (f"Satellite '{self.name}':\n"
                f"\tMass: {self.mass} kg\n"
                f"\tPosition: {pos_str}\n"
                f"\tVelocity: {vel_str}")


    def handleIndex(self, gap: int):
        """
        Handles the circular indexing of the position history.
        Args:
            gap (int): The number of indices to move forward.
        """

        if self.firstIndex > self.lastIndex:
            self.firstIndex += gap
            self.firstIndex %= self.maxIndex

        self.lastIndex += gap
        if self.lastIndex >= self.maxIndex:
            self.lastIndex %= self.maxIndex
            self.firstIndex = self.lastIndex + 1

    def getHistory(self) -> NDArray[np.float64]:
        """
        Returns the position history of the satellite.

        Returns:
            NDArray[np.float64]: The position history of the satellite.
        """

        if self.firstIndex <= self.lastIndex:
            history = self.position_history[self.firstIndex:self.lastIndex]
        else:
            history = np.concatenate((self.position_history[self.firstIndex:], self.position_history[:self.lastIndex]), axis=0)
        return history


    def move(self):
        """
        Moves the satellite by updating its position based on its velocity.
        """

        self.position += self.velocity

        self.position_history[self.lastIndex] = self.position.copy()
        self.handleIndex(1)


def acceleration(sat1: Satellite, sat2: Satellite, G: float, dt: float):
    """
    Calculates the gravitational acceleration between two satellites, and updates their velocities.

    Args:
        sat1 (Satellite): The first satellite.
        sat2 (Satellite): The second satellite.
        G (float): Gravitational constant.
        dt (float): Time step.

    Raises:
        ValueError: If the distance between the two satellites is zero.
    """

    diff = sat1.position - sat2.position
    dist = np.linalg.norm(diff)

    if dist == 0:
        raise ValueError("Distance between satellites cannot be zero.")

    unit_vector = diff / dist

    acc1 = (-unit_vector * (G * sat2.mass)) / (dist ** 2)
    acc2 = (unit_vector * (G * sat1.mass)) / (dist ** 2)

    sat1.velocity += acc1 * dt
    sat2.velocity += acc2 * dt
