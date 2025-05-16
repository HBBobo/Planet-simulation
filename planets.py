import numpy as np
from numpy.typing import NDArray
import matplotlib
import matplotlib.pyplot as plt


class Planet:

    name: str
    mass: float
    position: NDArray[np.float64]
    velocity: NDArray[np.float64]
    position_history: NDArray[np.float64]
    firstIndex: int                       # First index for storing data
    lastIndex: int                        # Next free index for storing data
    maxIndex: int                         # Number of stored data points
    marker: matplotlib.lines.Line2D
    line: matplotlib.lines.Line2D

    def __init__(self, axis: matplotlib.axes.Axes, name: str, mass: float, pos: NDArray[np.float64], vel: NDArray[np.float64], datapoints: int = 1024):
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

        self.marker = axis.plot([], [], 'o', markersize=6)[0]
        self.marker.set_label(self.name)

        self.line = axis.plot([], [], '-')[0]
        self.line.set_color(self.marker.get_color())


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


    def move(self, dt: float):
        """
        Moves the satellite by updating its position based on its velocity.

        Args:
            dt (float): Time step for the movement.
        """

        self.position += self.velocity * dt

        self.position_history[self.lastIndex] = self.position.copy()
        self.handleIndex(1)


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
            "velocity": self.velocity.tolist()
        }


def acceleration(sat1: Planet, sat2: Planet, G: float, dt: float):
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

    dist_square = dist ** 2

    acc1 = (-unit_vector * (G * sat2.mass)) / (dist_square)
    acc2 = ( unit_vector * (G * sat1.mass)) / (dist_square)

    sat1.velocity += acc1 * dt
    sat2.velocity += acc2 * dt


#-----------------------------------------------------------------------------------


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


    def __len__(self):

        return len(self.planets)


def from_dict(axis: matplotlib.axes.Axes, data: dict) -> Planet:
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
        axis=axis
    )