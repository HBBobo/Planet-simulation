from .planet import Planet, planet_from_dict

import numpy as np
from numpy.typing import NDArray
import matplotlib
import matplotlib.pyplot as plt
import json

class Planets:
    
    planets: list[Planet]
    axis: matplotlib.axes.Axes

    pos_all: np.ndarray  # Shape: (N, 2)
    vel_all: np.ndarray  # Shape: (N, 2)
    mass_all: np.ndarray # Shape: (N,)

    step: int
    step_per_show: int
    max_step: int
    G: float
    DT: float
    softening_factor_sq: float 

    def __init__(self, axis: matplotlib.axes.Axes, planets: list[Planet], sps: int, ms: int, G: float, dt: float, softening_factor: float = 1e-9):
        """
        Initializes the Planets class.
        
        Args:
            axis (matplotlib.axes.Axes): The axis on which to plot the planets.
            planets (list[Planet]): List of planets to simulate.
            sps (int): Steps per show.
            ms (int): Maximum steps.
            G (float): Gravitational constant.
            dt (float): Time step.
            softening_factor (float): Softening factor for gravitational force calculation.
        """

        self.planets = planets
        self.axis = axis

        self.step = 0
        self.step_per_show = sps
        self.max_step = ms

        self.G = G
        self.DT = dt
        self.softening_factor_sq = softening_factor ** 2
        self._initialize_arrays()


    def _initialize_arrays(self):
        """
        Helper to populate internal NumPy arrays from the list of Planet objects.
        """

        num_planets = len(self.planets)
        if num_planets == 0:
            self.pos_all = np.empty((0, 2), dtype=np.float64)
            self.vel_all = np.empty((0, 2), dtype=np.float64)
            self.mass_all = np.empty((0,), dtype=np.float64)
            return

        self.pos_all = np.array([p.position for p in self.planets], dtype=np.float64)
        self.vel_all = np.array([p.velocity for p in self.planets], dtype=np.float64)
        self.mass_all = np.array([p.mass for p in self.planets], dtype=np.float64)


    def _update_planet_objects(self):
        """
        Updates the planet objects with the current positions and velocities.
        """

        for i, planet_obj in enumerate(self.planets):
            planet_obj.update_display_data(self.pos_all[i], self.vel_all[i])


    def _calculate_accelerations(self) -> NDArray[np.float64]:
        """
        Calculates the accelerations of all planets in the system.
        """

        n_planets = len(self.planets)
        if n_planets < 2:
            return np.zeros_like(self.pos_all)

        # diff_x[i, j] = pos_all[i, 0] - pos_all[j, 0]
        # diff_y[i, j] = pos_all[i, 1] - pos_all[j, 1]
        diff_x = self.pos_all[:, np.newaxis, 0] - self.pos_all[np.newaxis, :, 0] # Shape: (N, N)
        diff_y = self.pos_all[:, np.newaxis, 1] - self.pos_all[np.newaxis, :, 1] # Shape: (N, N)
        
        # dist_sq[i,j] = |r_i - r_j|^2
        dist_sq = diff_x**2 + diff_y**2 + self.softening_factor_sq # Shape: (N, N)
        
        # set the diagonal to inf to avoid self-interaction
        np.fill_diagonal(dist_sq, np.inf)
        
        # Invers distance: 1 / |r_i - r_j|^3 = (dist_sq)^(-3/2)
        inv_dist_cubed = dist_sq**(-1.5) # Shape: (N, N)

        # a_i = sum_{j!=i} G * m_j * (r_j - r_i) / |r_i - r_j|^3
        # Broadcasting mass_all to (N, 1) to match the shape of diff_x and diff_y
        ax = self.G * np.sum(-diff_x * self.mass_all[np.newaxis, :] * inv_dist_cubed, axis=1) # Shape: (N,)
        ay = self.G * np.sum(-diff_y * self.mass_all[np.newaxis, :] * inv_dist_cubed, axis=1) # Shape: (N,)
        
        return np.stack((ax, ay), axis=1) # Shape: (N, 2)


    def _move_planets(self, accelerations: np.ndarray):
        """
        Updates the positions and velocities of all planets in the system.
        """
        self.vel_all += accelerations * self.DT
        self.pos_all += self.vel_all * self.DT


    def step_forward(self) -> int:
        """
        Steps forward in the simulation.
        Returns: 0 (running), 1 (show frame), 2 (finished).
        """
        if self.step >= self.max_step:
            return 2
        
        accelerations = self._calculate_accelerations()
        self._move_planets(accelerations)
        self._update_planet_objects()

        self.step += 1
        if self.step % self.step_per_show == 0:
            for planet_obj in self.planets:
                planet_obj.show()
            return 1

        if self.step >= self.max_step:
            return 2
            
        return 0


    def to_dict(self) -> dict:
        """
        Converts the Planets object to a dictionary.

        Returns:
            dict: A dictionary representation of the Planets object.
        """

        return {
            "version": "1.1",
            "planets": [p.to_dict() for p in self.planets],
            "sps": self.step_per_show,
            "ms": self.max_step,
            "G": self.G,
            "dt": self.DT,
            "softening_factor": np.sqrt(self.softening_factor_sq)
        }


    def save(self, path:str):
        """
        Saves the current state of the planets to a file.
        """

        with open(path, "w") as file:
            json.dump(self.to_dict(), file, indent=4)


    def __len__(self):
        return len(self.planets)

    def __getitem__(self, index: int) -> Planet:
        return self.planets[index]


def planets_from_dict(axis: matplotlib.axes.Axes, data: dict) -> list[Planet]:
    """
    Creates a list of Satellite objects from a list of dictionaries.

    Args:
        data (list[dict]): A list of dictionaries containing the satellites' attributes.

    Returns:
        list[Satellite]: A list of Satellite objects.
    """

    version = data.get("version", "0.0") # Default verzió, ha hiányzik
    planet_list = [planet_from_dict(axis, p_data) for p_data in data.get("planets", data)]

    if version == "0.0":
        return Planets(
            planets=planet_list,
            axis=axis,
            sps=200,
            ms=100000,
            G=6.67430e-11,
            dt=2.0,
            softening_factor=1e-9
        )
    elif version == "1.0":
         return Planets(
            planets=planet_list,
            axis=axis,
            sps=data["sps"],
            ms=data["ms"],
            G=data["G"],
            dt=data["dt"],
            softening_factor=1e-9
        )
    elif version == "1.1":
        return Planets(
            planets=planet_list,
            axis=axis,
            sps=data["sps"],
            ms=data["ms"],
            G=data["G"],
            dt=data["dt"],
            softening_factor=data.get("softening_factor", 1e-9)
        )
    else:
        raise ValueError(f"Unsupported data version: {version}")


def load(path: str, axis: matplotlib.axes.Axes) -> Planets:
    """
    Loads the planets from a file.

    Args:
        path (str): The path to the file containing the planets' attributes.
        axis (matplotlib.axes.Axes): The axis on which to plot the planets.

    Returns:
        Planets: A Planets object.
    """
    
    with open(path, "r") as file:
        data = json.load(file)
    return planets_from_dict(axis, data)