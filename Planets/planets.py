from .planet import Planet, acceleration, planet_from_dict

import numpy as np
from numpy.typing import NDArray
import matplotlib
import matplotlib.pyplot as plt
import json

class Planets:
    
    planets: list[Planet]
    axis: matplotlib.axes.Axes
    step: int
    step_per_show: int
    max_step: int
    G: float
    DT: float

    def __init__(self, axis: matplotlib.axes.Axes, planets: list[Planet], sps: int, ms: int, G: float, dt: float):
        """
        Initializes the Planets class.
        
        Args:
            axis (matplotlib.axes.Axes): The axis on which to plot the planets.
            planets (list[Planet]): List of planets to simulate.
            sps (int): Steps per show.
            ms (int): Maximum steps.
            G (float): Gravitational constant.
            dt (float): Time step.
        """

        self.planets = planets
        self.axis = axis

        self.step = 0
        self.step_per_show = sps
        self.max_step = ms

        self.G = G
        self.DT = dt


    def add(self, planet: Planet):
        """
        Adds a planet to the list.

        Args:
            planet (Planet): The planet to add.
        """

        self.planets.append(planet)


    def get(self, index: int) -> Planet:
        """
        Gets a planet by index.

        Args:
            index (int): The index of the planet to get.

        Returns:
            planet (Planet): The planet at the specified index.
        """

        return self.planets[index]


    def remove(self, planet: Planet):
        """
        Removes a planet from the list.

        Args:
            planet (Planet): The planet to remove.
        """

        self.planets.remove(planet)


    def acceleration(self):
        """
        Calculates the gravitational acceleration between all planets in the list.
        """

        for i in range(len(self.planets)):
            for j in range(i + 1, len(self.planets)):

                try:
                    acceleration(self.planets[i], self.planets[j], self.G, self.DT)

                except ValueError as e:
                    print(f"Pair ({self.planets[i].name}, {self.planets[j].name}): ValueError: {e}")

                except Exception as e:
                    print(f"Pair ({self.planets[i].name}, {self.planets[j].name}): An unexpected error occurred: {e}")
                    raise


    def move(self):
        """
        Moves all planets in the list.
        """
        for planet in self.planets:
            planet.move(self.DT)


    def step_forward(self) -> int:
        """
        Steps forward in the simulation.

        Returns:
            int: 0 if the simulation is still running, 1 if the simulation is paused for showing, 2 if the simulation is finished.
        """

        self.acceleration()
        self.move()

        if self.step % self.step_per_show == 0:
            for planet in self.planets:
                planet.show()

            self.step += 1
            return 1

        self.step += 1

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
            "version": "1.0",
            "planets": [planet.to_dict() for planet in self.planets],
            "sps": self.step_per_show,
            "ms": self.max_step,
            "G": self.G,
            "dt": self.DT
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

    def __setitem__(self, index: int, planet: Planet):
        if index < 0 or index >= len(self.planets):
            raise IndexError("Index out of range.")
        self.planets[index] = planet

    def __delitem__(self, index: int):
        if index < 0 or index >= len(self.planets):
            raise IndexError("Index out of range.")
        del self.planets[index]



def planets_from_dict(axis: matplotlib.axes.Axes, data: dict) -> list[Planet]:
    """
    Creates a list of Satellite objects from a list of dictionaries.

    Args:
        data (list[dict]): A list of dictionaries containing the satellites' attributes.

    Returns:
        list[Satellite]: A list of Satellite objects.
    """

    if "version" in data:
        version = data["version"]

    else:
        version = "0.0"

    if version == "0.0":
        return Planets(planets = [planet_from_dict(axis, planet) for planet in data],
                       axis = axis,
                       sps = 200,
                       ms = 100000,
                       G = 6.67430e-11,
                       dt = 2.0)

    elif version == "1.0":
        return Planets(planets=[planet_from_dict(axis, planet) for planet in data["planets"]],
                       axis=axis,
                       sps=data["sps"],
                       ms=data["ms"],
                       G=data["G"],
                       dt=data["dt"])


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