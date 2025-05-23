from utils.circular_buffer import CircularBuffer

import numpy as np
from numpy.typing import NDArray
import matplotlib
import matplotlib.pyplot as plt
import json

class Planet:

    name: str
    mass: float
    position: NDArray[np.float64]
    velocity: NDArray[np.float64]
    position_history: CircularBuffer
    marker: matplotlib.lines.Line2D
    line: matplotlib.lines.Line2D

    def __init__(self, axis: matplotlib.axes.Axes, name: str, mass: float, pos: NDArray[np.float64], vel: NDArray[np.float64], datapoints: int = 65536):
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

        self.position_history = CircularBuffer(datapoints, 2)
        self.position_history.append(self.position.copy())

        self.marker = axis.plot([], [], 'o', markersize=6)[0]
        self.marker.set_label(self.name)

        self.line = axis.plot([], [], '-')[0]
        self.line.set_color(self.marker.get_color())


    def __str__(self):
        """
        Representation of the Satellite object

        Returns:
            str: A string representation of the Satellite object.
        """

        pos_str = f"[{self.position[0]:.2e}, {self.position[1]:.2e}] m"
        vel_str = f"[{self.velocity[0]:.2e}, {self.velocity[1]:.2e}] m/s"

        return (f"Planet '{self.name}': Mass: {self.mass:.2e} kg, "
                f"Pos: {pos_str}, Vel: {vel_str}")


    def getHistory(self) -> NDArray[np.float64]:
        """
        Returns the position history of the satellite.

        Returns:
            NDArray[np.float64]: The position history of the satellite.
        """

        return self.position_history.get()


    def update_display_data(self, new_position: NDArray[np.float64], new_velocity: NDArray[np.float64]):
        """
        Updates the display data of the satellite.
        """
        self.position = new_position.copy()
        self.velocity = new_velocity.copy()
        self.position_history.append(self.position)


    def show(self):
        self.marker.set_data([self.position[0]], [self.position[1]])
        data = self.getHistory()
        self.line.set_data(data[:, 0], data[:, 1])
        

    def to_dict(self) -> dict:
        """
        Converts the Satellite object to a dictionary.

        Returns:
            dict: A dictionary representation of the Satellite object.
        """

        return {
            "name": self.name,
            "mass": self.mass,
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "datapoints": self.position_history.size
        }


def planet_from_dict(axis: matplotlib.axes.Axes, data: dict) -> Planet:
    """
    Creates a Satellite object from a dictionary.
    Args:
        data (dict): A dictionary containing the satellite's attributes.
    Returns:
        Satellite: A Satellite object.
    """
    return Planet(
        name=data["name"],
        mass=data["mass"],
        pos=np.array(data["position"], dtype=np.float64),
        vel=np.array(data["velocity"], dtype=np.float64),
        axis=axis,
        datapoints=data.get("datapoints", 65536)
    )